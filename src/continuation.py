
def load_continuation_sentences(file):


def continue_sentence(sentence, lstm, session):

    feed_dict = {lstm.input_sentence: sentences_batch}

    fetches = [lstm.loss, summaries_merged, train_step, global_step]
    loss, summary, _, step = session.run(fetches, feed_dict)

    train_writer.add_summary(summary, step)
    raise NotImplementedError
