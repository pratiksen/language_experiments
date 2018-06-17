from experiment import ex


if __name__ == '__main__':
    ex.run(config_updates={'epochs':10, 'verbose':2,
                           'params': {'max_sentence_len':None,
                                      'units':[128],
                                      'rnn_params':{'recurrent_dropout':0.5}
                                      }
                           }
           )