from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import asyncio
import uuid
from pathlib import Path
import io

from ...core.pipeline import FluxPipeline
from ...utils.logging import get_logger
from ...config.settings import OUTPUT_DIR

router = APIRouter(prefix="/api/v1", tags=["generation"])
logger = get_logger(__name__)

pipeline = FluxPipeline()

class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation", min_length=1, max_length=512)
    negative_prompt: Optional[str] = Field(None, description="Negative prompt to avoid certain features")
    width: int = Field(1024, ge=256, le=2048, description="Image width")
    height: int = Field(1024, ge=256, le=2048, description="Image height")
    num_inference_steps: int = Field(4, ge=1, le=50, description="Number of denoising steps")
    guidance_scale: float = Field(3.5, ge=1.0, le=20.0, description="Guidance scale for classifier-free guidance")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    num_images: int = Field(1, ge=1, le=4, description="Number of images to generate")

class GenerationResponse(BaseModel):
    job_id: str
    status: str
    image_urls: Optional[List[str]] = None
    message: Optional[str] = None

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float  # 0.0 to 1.0
    image_urls: Optional[List[str]] = None
    error: Optional[str] = None

# In-memory job storage (use Redis in production)
jobs = {}

@router.post("/generate", response_model=GenerationResponse)
async def generate_image(
    request: GenerationRequest,
    background_tasks: BackgroundTasks
):
    """Generate images from text prompt
    
    **Example Request:**
    ```json
    {
        "prompt": "a beautiful sunset over mountains, highly detailed, 4k",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 4,
        "guidance_scale": 3.5
    }
    ```
    
    **Returns:**
    - job_id: Unique identifier for tracking generation
    - status: Current status of the job
    - image_urls: URLs to download generated images (when completed)
    """
    try:
        job_id = str(uuid.uuid4())
        
        jobs[job_id] = {
            "status": "pending",
            "progress": 0.0,
            "image_urls": [],
            "error": None
        }
        
        background_tasks.add_task(
            process_generation,
            job_id,
            request
        )
        
        logger.info(f"Created generation job {job_id} for prompt: {request.prompt[:50]}...")
        
        return GenerationResponse(
            job_id=job_id,
            status="pending",
            message="Generation job created successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to create generation job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_generation(job_id: str, request: GenerationRequest):
    """Background task to process image generation"""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 0.1
        
        output_dir = Path(OUTPUT_DIR) / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_urls = []
        
        for i in range(request.num_images):
            jobs[job_id]["progress"] = 0.1 + (i / request.num_images) * 0.8
            
            output_path = output_dir / f"image_{i}.png"
            
            image_bytes = pipeline(
                request.prompt,
                output_path=str(output_path),
                width=request.width,
                height=request.height,
                steps=request.num_inference_steps,
                guidance=request.guidance_scale
            )
            
            image_urls.append(f"/api/v1/images/{job_id}/image_{i}.png")
            
            logger.info(f"Generated image {i+1}/{request.num_images} for job {job_id}")
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["image_urls"] = image_urls
        
        logger.info(f"Completed generation job {job_id}")
    
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        logger.error(f"Generation job {job_id} failed: {str(e)}")

@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of a generation job
    
    **Returns:**
    - status: pending, processing, completed, or failed
    - progress: 0.0 to 1.0
    - image_urls: List of URLs when completed
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        image_urls=job.get("image_urls"),
        error=job.get("error")
    )

@router.get("/images/{job_id}/{filename}")
async def get_generated_image(job_id: str, filename: str):
    """Download a generated image
    
    **Parameters:**
    - job_id: Job identifier
    - filename: Image filename
    
    **Returns:**
    - PNG image file
    """
    image_path = Path(OUTPUT_DIR) / job_id / filename
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(
        image_path,
        media_type="image/png",
        filename=filename
    )

@router.post("/generate/sync")
async def generate_image_sync(request: GenerationRequest):
    """Synchronous image generation (blocks until complete)
    
    **Use this for:**
    - Single image generation
    - When you need immediate results
    - Testing and development
    
    **For production, use async /generate endpoint**
    """
    try:
        job_id = str(uuid.uuid4())
        output_dir = Path(OUTPUT_DIR) / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "image_0.png"
        
        image_bytes = pipeline(
            request.prompt,
            output_path=str(output_path),
            width=request.width,
            height=request.height,
            steps=request.num_inference_steps,
            guidance=request.guidance_scale
        )
        
        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=generated_{job_id}.png"}
        )
    
    except Exception as e:
        logger.error(f"Sync generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a generation job and its outputs
    
    **Cleanup:**
    - Removes job from memory
    - Deletes generated images
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    output_dir = Path(OUTPUT_DIR) / job_id
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    
    del jobs[job_id]
    
    logger.info(f"Deleted job {job_id}")
    
    return {"message": "Job deleted successfully"}

@router.get("/stats")
async def get_generation_stats():
    """Get generation statistics
    
    **Returns:**
    - Total jobs
    - Jobs by status
    - Average generation time
    """
    total_jobs = len(jobs)
    
    status_counts = {
        "pending": 0,
        "processing": 0,
        "completed": 0,
        "failed": 0
    }
    
    for job in jobs.values():
        status = job.get("status", "unknown")
        if status in status_counts:
            status_counts[status] += 1
    
    return {
        "total_jobs": total_jobs,
        "by_status": status_counts,
        "active_jobs": status_counts["pending"] + status_counts["processing"]
    }
