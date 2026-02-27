from sklearn.pipeline import Pipeline
from src.models import tokenizer_stemmer_es, get_model


def test_tokenizer_stemmer_es_output():
    """
    Test that the tokenizer correctly stems
    """
    sample_text = "Estamos trabajando en una solución excelente."  
    # Remember that input data can be parametrized with pytest.fixture. Try it your self!
    tokens = tokenizer_stemmer_es(sample_text)

    # Should return a list of str
    assert ...  # list
    assert ...  # any element is str

    # Ensure some stemmed tokens are returned (any elements)
    assert ...


def test_tokenizer_stemmer_es_remove_punct():
    """
    Test that the tokenizer correctly removes punctuation.
    """

    sample_text = "Estamos trabajando en una solución excelente."
    tokens = tokenizer_stemmer_es(sample_text)

    # Should return stemmed words without punctuation
    assert ...  # "." is not in returned list


def test_tokenizer_stemmer_es_example():
    """
    Test that the tokenizer correctly perform stemming.
    """

    sample_text = "Estamos trabajando en una solución excelente."
    tokens = tokenizer_stemmer_es(sample_text)

    assert ...  # get by hand a stemmed term example and check that it works


def test_get_model_pipeline():
    """
    Test that get_model returns a sklearn Pipeline with a fte and a clf steps
    """
    skl_pl = get_model(min_df=1, max_df=1.0, max_features=1000)
    
    assert ...  # is a Pipeline instance
    step_names = [name for name, _ in skl_pl.steps]
    assert ...  # fte name exists
    assert ...  # clf name exists