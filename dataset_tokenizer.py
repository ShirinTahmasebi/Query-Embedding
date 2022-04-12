# For having a SQL-based tokenizer, a PERL module should be called. To do so, the following commands need to be executed in terminal:
# curl -L https://cpanmin.us | perl - --sudo App::cpanminus
# cpanm SQL::Tokenizer

# For using the mentioned PERL module and calling it in Python code, `helper_tokenization´ is used. This helper is copied from:
# https://github.com/boriel/perlfunc
from constants import Constants
from dataset_models_bombay import DatasetBombay


def create_and_save_tokenizer_script():
    perl_script = """
    sub callee {
      my $query = $_[0];
    
      use SQL::Tokenizer qw(tokenize_sql);
      my @tokens= SQL::Tokenizer->tokenize($query);
    
      @tokens= tokenize_sql($query);
    
      return (@tokens)
    }
    
    1;
    """

    f = open(Constants.PATH_BASE + "/tokenizer_script.pl", "w")
    f.write(perl_script)
    f.close()


from helper_tokenization import perl5lib, perlfunc, perlreq


@perlfunc
@perlreq(Constants.PATH_BASE + '/tokenizer_script.pl')
def callee(query):
    pass  # Empty body


if __name__ == '__main__':
    create_and_save_tokenizer_script()

    dataset = DatasetBombay()

    df = dataset.process()
    df['tokens'] = [callee(query) for query in df['full_query']]
    print('Tokenization was successful!')

    df.to_csv(dataset.get_path_labeled_tokenized_dataset(), index=False)
    print('Tokenized dataframe is saved!')
