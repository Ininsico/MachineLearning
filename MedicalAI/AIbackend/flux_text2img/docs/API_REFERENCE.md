# API Reference

Complete API documentation for FLUX Text-to-Image Generation System.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

All API requests require authentication via Bearer token:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.flux-t2i.com/api/v1/generate
```

---

## Endpoints

### 1. Generate Image (Async)

**POST** `/api/v1/generate`

Generate images asynchronously. Returns immediately with a job ID for tracking.

#### Request Body

```json
{
  "prompt": "a beautiful sunset over mountains, highly detailed, 4k",
  "negative_prompt": "blurry, low quality, distorted",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 4,
  "guidance_scale": 3.5,
  "seed": 42,
  "num_images": 1
}
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Text description of desired image (1-512 chars) |
| `negative_prompt` | string | No | null | What to avoid in generation |
| `width` | integer | No | 1024 | Image width (256-2048) |
| `height` | integer | No | 1024 | Image height (256-2048) |
| `num_inference_steps` | integer | No | 4 | Denoising steps (1-50, higher=better quality) |
| `guidance_scale` | float | No | 3.5 | CFG scale (1.0-20.0, higher=more prompt adherence) |
| `seed` | integer | No | random | Random seed for reproducibility |
| `num_images` | integer | No | 1 | Number of images to generate (1-4) |

#### Response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Generation job created successfully"
}
```

#### Example

```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cat wearing a space suit, digital art",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 4
  }'
```

---

### 2. Check Job Status

**GET** `/api/v1/jobs/{job_id}`

Check the status of a generation job.

#### Response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 1.0,
  "image_urls": [
    "/api/v1/images/550e8400-e29b-41d4-a716-446655440000/image_0.png"
  ],
  "error": null
}
```

#### Status Values

- `pending`: Job queued, not started
- `processing`: Currently generating
- `completed`: Generation finished successfully
- `failed`: Generation failed (check `error` field)

#### Example

```bash
curl "http://localhost:8000/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000"
```

---

### 3. Download Generated Image

**GET** `/api/v1/images/{job_id}/{filename}`

Download a generated image.

#### Response

Returns PNG image file.

#### Example

```bash
curl "http://localhost:8000/api/v1/images/550e8400-e29b-41d4-a716-446655440000/image_0.png" \
  --output generated_image.png
```

---

### 4. Generate Image (Sync)

**POST** `/api/v1/generate/sync`

Generate image synchronously. Blocks until generation completes.

⚠️ **Warning**: Only use for single images. For production, use async endpoint.

#### Request Body

Same as async `/generate` endpoint.

#### Response

Returns PNG image directly.

#### Example

```bash
curl -X POST "http://localhost:8000/api/v1/generate/sync" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a dog in a park",
    "width": 512,
    "height": 512
  }' \
  --output image.png
```

---

### 5. Delete Job

**DELETE** `/api/v1/jobs/{job_id}`

Delete a job and its generated images.

#### Response

```json
{
  "message": "Job deleted successfully"
}
```

#### Example

```bash
curl -X DELETE "http://localhost:8000/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000"
```

---

### 6. Get Statistics

**GET** `/api/v1/stats`

Get generation statistics.

#### Response

```json
{
  "total_jobs": 150,
  "by_status": {
    "pending": 5,
    "processing": 2,
    "completed": 140,
    "failed": 3
  },
  "active_jobs": 7
}
```

#### Example

```bash
curl "http://localhost:8000/api/v1/stats"
```

---

## Error Responses

All endpoints return standard HTTP status codes:

### Success Codes

- `200 OK`: Request successful
- `201 Created`: Resource created

### Error Codes

- `400 Bad Request`: Invalid parameters
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## Rate Limiting

- **Free tier**: 10 requests/minute
- **Pro tier**: 100 requests/minute
- **Enterprise**: Unlimited

Rate limit headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640000000
```

---

## Best Practices

### 1. Prompt Engineering

**Good prompts:**
```
"a majestic lion in the savanna, golden hour lighting, highly detailed, 4k, professional photography"
```

**Bad prompts:**
```
"lion"
```

### 2. Parameter Tuning

| Use Case | Steps | Guidance | Size |
|----------|-------|----------|------|
| Quick draft | 4 | 3.5 | 512x512 |
| Standard quality | 8 | 5.0 | 1024x1024 |
| High quality | 20 | 7.5 | 1024x1024 |
| Maximum quality | 50 | 10.0 | 2048x2048 |

### 3. Async vs Sync

- **Use Async** for:
  - Multiple images
  - High-resolution generation
  - Production applications
  
- **Use Sync** for:
  - Quick tests
  - Single low-res images
  - Development

### 4. Error Handling

```python
import requests
import time

def generate_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/api/v1/generate",
                json={"prompt": prompt}
            )
            response.raise_for_status()
            
            job_id = response.json()["job_id"]
            
            # Poll for completion
            while True:
                status_response = requests.get(
                    f"http://localhost:8000/api/v1/jobs/{job_id}"
                )
                status = status_response.json()
                
                if status["status"] == "completed":
                    return status["image_urls"]
                elif status["status"] == "failed":
                    raise Exception(status["error"])
                
                time.sleep(2)
        
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(5)
```

---

## Webhooks

Configure webhooks to receive notifications when jobs complete:

**POST** `/api/v1/webhooks`

```json
{
  "url": "https://your-app.com/webhook",
  "events": ["generation.completed", "generation.failed"]
}
```

Webhook payload:

```json
{
  "event": "generation.completed",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "image_urls": ["/api/v1/images/..."]
  }
}
```

---

## SDKs

Official SDKs available:

- **Python**: `pip install flux-t2i-sdk`
- **JavaScript**: `npm install @flux/t2i-sdk`
- **Go**: `go get github.com/flux/t2i-sdk`

### Python SDK Example

```python
from flux_t2i import FluxClient

client = FluxClient(api_key="YOUR_API_KEY")

# Generate image
job = client.generate(
    prompt="a beautiful landscape",
    width=1024,
    height=1024
)

# Wait for completion
job.wait()

# Download image
job.download("output.png")
```

---

## Support

- **Documentation**: https://docs.flux-t2i.com
- **API Status**: https://status.flux-t2i.com
- **Discord**: https://discord.gg/flux-t2i
- **Email**: support@flux-t2i.com
