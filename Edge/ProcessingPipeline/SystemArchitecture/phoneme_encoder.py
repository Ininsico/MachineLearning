from phonemizer import phonemize

class PhonemeEncoder:
    def __init__(self, language='en-us'):
        self.language = language
        self.phoneme2id = {}
        self.id2phoneme = {}

    def __call__(self, text):
        phonemes = phonemize(
            text,
            language=self.language,
            backend='espeak',
            strip=True,
            preserve_punctuation=True,
            njobs=1
        )
        phonemes = phonemes.split()
        ids = []
        for p in phonemes:
            if p not in self.phoneme2id:
                idx = len(self.phoneme2id)
                self.phoneme2id[p] = idx
                self.id2phoneme[idx] = p
            ids.append(self.phoneme2id[p])
        return ids