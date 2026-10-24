import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from ougd.prepare import pack_documents

class Tokenizer:
    eos_token_id = 0
    def encode(self, text, **kwargs):
        return [int(c) for c in text]

class DataTests(unittest.TestCase):
    def test_document_split_and_boundary_predictions(self):
        docs = [{'id':str(i),'text':str(i%9+1)*10} for i in range(20)]
        docs.append(dict(id='duplicate',text=docs[0]['text']))
        with tempfile.TemporaryDirectory() as d:
            counts = pack_documents(docs,Tokenizer(),4,16,d,.5)
            manifest = [json.loads(x) for x in (Path(d)/'documents.jsonl').read_text().splitlines()]
            self.assertEqual(manifest[0]['split'],manifest[-1]['split'])
            for split in counts:
                rows = np.load(Path(d)/f'{split}.npy')
                self.assertEqual(counts[split],len(rows)*4)
                expected = []
                for doc,entry in zip(docs,manifest):
                    if entry['split']==split: expected.extend(Tokenizer().encode(doc['text'])+[0])
                reconstructed = [rows[0,0]]+rows[:,1:].flatten().tolist() if len(rows) else []
                self.assertEqual(reconstructed,expected[:len(reconstructed)])
                if len(rows)>1: np.testing.assert_array_equal(rows[:-1,-1],rows[1:,0])

if __name__=='__main__': unittest.main()
