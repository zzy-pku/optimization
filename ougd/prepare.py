"""Offline JSONL/FineWeb preparation; document-level deterministic holdout before packing."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def pack_documents(documents, tokenizer, context, vocab, directory, validation_fraction=.01):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    buffers = {'train': [], 'validation': []}
    counts = dict.fromkeys(buffers, 0)
    files = {k: (directory/f'{k}.raw').open('wb') for k in buffers}
    manifest = (directory/'documents.jsonl').open('w')
    try:
        for doc in documents:
            text = doc['text']
            # Content hash keeps duplicate documents in the same split.
            digest = hashlib.sha256(text.encode()).hexdigest()
            split = 'validation' if int(digest[:16],16)/2**64 < validation_fraction else 'train'
            ids = tokenizer.encode(text, add_special_tokens=False)
            if tokenizer.eos_token_id is None: raise ValueError('tokenizer requires EOS')
            ids.append(tokenizer.eos_token_id)
            if min(ids) < 0 or max(ids) >= vocab: raise ValueError('tokenizer exceeds model vocabulary')
            manifest.write(json.dumps(dict(document_id=str(doc.get('id',digest)),sha256=digest,
                split=split, tokens=len(ids), stream_offset=counts[split]+len(buffers[split])))+'\n')
            buffers[split].extend(ids)
            while len(buffers[split]) >= context+1:
                row = buffers[split][:context+1]
                files[split].write(np.asarray(row,dtype=np.int32).tobytes())
                # Overlap the boundary token so each prediction in the stream is counted once.
                del buffers[split][:context]
                counts[split] += context
    finally:
        manifest.close()
        for f in files.values(): f.close()
    for split in buffers:
        raw = directory/f'{split}.raw'
        rows = raw.stat().st_size//(4*(context+1))
        out = np.lib.format.open_memmap(directory/f'{split}.npy',mode='w+',dtype=np.int32,shape=(rows,context+1))
        if rows:
            source = np.memmap(raw,dtype=np.int32,mode='r',shape=(rows,context+1))
            for i in range(0,rows,4096): out[i:i+4096] = source[i:i+4096]
            del source
        out.flush()
        del out
        raw.unlink()
    return counts

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--jsonl')
    p.add_argument('--fineweb',action='store_true')
    p.add_argument('--documents',type=int,default=100000)
    p.add_argument('--tokenizer',required=True,help='local tokenizer directory or HF model ID; select explicitly')
    p.add_argument('--tokenizer-revision',default='main')
    p.add_argument('--data-revision',default='main')
    p.add_argument('--context',type=int,default=2048)
    p.add_argument('--vocab',type=int,default=32000)
    p.add_argument('--validation-fraction',type=float,default=.01)
    p.add_argument('--output',required=True)
    args = p.parse_args()
    if bool(args.jsonl) == args.fineweb: raise ValueError('select exactly one source')
    if not 0 < args.validation_fraction < 1 or args.context < 1: raise ValueError('invalid split or context')
    dest = Path(args.output)
    if dest.exists() and any(dest.iterdir()): raise ValueError('output must be empty')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,revision=args.tokenizer_revision)
    if args.fineweb:
        from datasets import load_dataset
        ds = load_dataset('HuggingFaceFW/fineweb-edu','sample-10BT',split='train',streaming=True,revision=args.data_revision)
        docs = ds.take(args.documents)
    else:
        def read():
            with open(args.jsonl) as f:
                for line in f: yield json.loads(line)
        docs = read()
    counts = pack_documents(docs,tokenizer,args.context,args.vocab,args.output,args.validation_fraction)
    tokenizer.save_pretrained(dest/'tokenizer')
    from .data import sha256
    metadata = dict(args=vars(args),prediction_tokens=counts,
        token_files={s:sha256(dest/f'{s}.npy') for s in counts},
        tokenizer_files={f.name:sha256(f) for f in (dest/'tokenizer').iterdir() if f.is_file()},
        document_manifest_sha256=sha256(dest/'documents.jsonl'))
    (dest/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))

if __name__=='__main__': main()
