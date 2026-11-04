import numpy as np
from app.engine import clean, chunk_pages, cosines, preview_vector, extract

def test_clean_preserves_words():
    assert clean('Hello   world\n again') == 'Hello world again'

def test_chunk_boundaries_and_overlap():
    chunks=chunk_pages([{'page':3,'text':' '.join(str(i) for i in range(14))}],size=6,overlap=2)
    assert [(c['start_word'],c['end_word']) for c in chunks] == [(0,6),(4,10),(8,14)]
    assert chunks[1]['text'].startswith('4 5')

def test_cosines():
    result=cosines(np.array([1,0]),np.array([[1,0],[0,1],[-1,0]]))
    np.testing.assert_allclose(result,[1,0,-1])

def test_preview():
    assert preview_vector(np.array([.123456789,-.5]),count=1)==[.123457]

def test_txt():
    assert extract('hello.txt',b'Example text')[0]['text']=='Example text'
