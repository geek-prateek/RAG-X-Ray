"""End-to-end API smoke test using a deterministic test encoder, no model downloads."""
import time
import numpy as np
from fastapi.testclient import TestClient
from app import engine, main, storage

class FakeCollection:
    def __init__(self): self.values={}
    def upsert(self, ids, embeddings, documents, metadatas):
        for args in zip(ids,embeddings,documents,metadatas):self.values[args[0]]=args[1:]
    def get(self, ids, include):
        ids=[i for i in ids if i in self.values]
        return {'ids': ids, 'embeddings':[self.values[i][0] for i in ids],
                'documents':[self.values[i][1] for i in ids], 'metadatas':[self.values[i][2] for i in ids]}
    def delete(self, ids):
        for i in ids:self.values.pop(i,None)

def test_upload_and_question(monkeypatch,tmp_path):
    fake=FakeCollection()
    monkeypatch.setattr(storage,'DATA',tmp_path)
    monkeypatch.setattr(storage,'collection',lambda:fake)
    monkeypatch.setattr(engine,'embed',lambda texts: np.array([[1.,0.] if 'leave' in t.lower() else [0.,1.] for t in texts]))
    monkeypatch.setattr(engine,'ollama_answer',lambda q,ctx: (' '.join(c['text'] for c in ctx),'test_extractive'))
    with TestClient(main.app) as client:
        r=client.post('/api/documents',files={'file':('policy.txt',b'Employees request leave on the HR portal.\nThey can reset passwords on the login page.','text/plain')})
        assert r.status_code==202
        doc_id=r.json()['document_id']
        for _ in range(100):
            d=client.get('/api/documents/'+doc_id).json()
            if d['status']!='processing' and d['status']!='queued':break
            time.sleep(.02)
        assert d['status']=='ready',d.get('error')
        assert len(d['stages'])==6
        assert d['stages'][4]['after']['shape']==[1,2]
        out=client.post('/api/ask',json={'question':'leave','document_ids':[doc_id],'top_k':1})
        assert out.status_code==200,out.text
        response=out.json()
        assert response['ranked'][0]['score']==1.
        assert response['steps'][1]['after']['values']==[1.,0.]
        assert response['sources'][0]['filename']=='policy.txt'
        assert client.delete('/api/documents/'+doc_id).status_code==200
        assert client.get('/api/documents/'+doc_id).status_code==404
