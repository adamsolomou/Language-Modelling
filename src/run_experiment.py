import tensorflow as tf
from data_processing import DataProcessing
from model import LSTMCell
from utils import train_batch, dev_step, load_embedding, continue_sentences
from data_processing import get_batches
import os
import time
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)  # suppress some deprecation warnings

device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')

def run_experiment(experiment_type, data_folder, save_model_folder, save_results_folder):
    """
    Runs experiments and saves results

    Parameters
    ----------
    experiment_type
    data_folder
    save_model_folder
    save_results_folder
    """
    def set_experiment_variables(hidden_state_size=512, down_project_size=None, load_embeddings=False):
        tf.flags.DEFINE_integer("hidden_state_size", hidden_state_size, "hidden state size (default 512)")
        tf.flags.DEFINE_integer("down_project_size", down_project_size,
                                "Down projection size. Should be used with a hidden_state_size of 1024 (default None)")
        tf.flags.DEFINE_boolean("load_embeddings", load_embeddings,
                                "Whether to use pretrained embeddings or not (default False)")
    if experiment_type == 'A':
        set_experiment_variables(512, None, False)
    elif experiment_type == 'B':
        set_experiment_variables(512, None, True)
    elif experiment_type == 'C':
        set_experiment_variables(1024, 512, False)

    print("\nExperiment Arguments:")
    for key in FLAGS.flag_values_dict():
        if key == 'f':
            continue
        print("{:<22}: {}".format(key.upper(), FLAGS[key].value))
    print(" ")

    data_processing = DataProcessing(FLAGS.sentence_length, FLAGS.max_vocabulary_size)
    data_processing.preprocess_dataset(data_folder, 'sentences.train', 'train_corpus')
    data_processing.preprocess_dataset(data_folder, 'sentences.eval', 'validation_corpus')
    data_processing.preprocess_dataset(data_folder, 'sentences_test.txt', 'test_corpus')
    data_processing.preprocess_dataset(data_folder, 'sentences.continuation', 'continuation_corpus',
                                       pad_to_sentence_length=False)

    print(f'Number of train sentences is \t\t{len(data_processing.train_corpus)}')
    print(f'Number of validation sentences is \t{len(data_processing.validation_corpus)}')
    print(f'Number of test sentences is \t\t{len(data_processing.test_corpus)}')
    print(f'Number of continuation sentences is \t{len(data_processing.continuation_corpus)}')
    print(" ")

    best_perplexity = None
    best_model = None

    with tf.Graph().as_default():
        with tf.Session() as session:
            # Create a variable to contain a counter for the global training step.
            global_step = tf.Variable(1, name='global_step', trainable=False)

            lstm = LSTMCell(FLAGS.embedding_size, FLAGS.hidden_state_size, FLAGS.sentence_length,
                            FLAGS.max_vocabulary_size, down_project_size=FLAGS.down_project_size,
                            pad_symbol=data_processing.vocab['<pad>'])

            if FLAGS.load_embeddings:
                load_embedding(session, data_processing.vocab, lstm.input_embeddings,
                               data_folder + '/wordembeddings-dim100.word2vec',
                               FLAGS.embedding_size, len(data_processing.vocab))

            ####
            # Set optimizer and crop all gradients to values [-5, 5]
            ####
            with tf.name_scope('train'):
                optimizer = tf.train.AdamOptimizer()
                gvs = optimizer.compute_gradients(lstm.loss)
                capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
                train_step = optimizer.apply_gradients(capped_gvs, global_step=global_step)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            session.run(tf.global_variables_initializer())
            summaries_merged = tf.summary.merge(lstm.summaries)

            ####
            # Create checkpoint directory
            ####
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(save_model_folder, "runs", timestamp))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            ####
            # Start training for the specified epochs
            ####
            print('Start training...')
            for epoch in range(FLAGS.num_epochs):
                for sentences_batch in get_batches(data_processing.train_corpus, batch_size=FLAGS.batch_size):
                    # run a single step
                    train_batch(sentences_batch, lstm, train_step, global_step, session, summaries_merged)

                current_step = tf.train.global_step(session, global_step)
                if current_step % FLAGS.checkpoint_every == 0:
                    perplexities = dev_step(get_batches(data_processing.validation_corpus, batch_size=FLAGS.batch_size,
                                                        do_shuffle=False), lstm, global_step, session)

                    average_perplexity = np.mean(perplexities)
                    model_name = "model_experiment-{}_epoch-{}_val-perplexity-{}".format(experiment_type, epoch + 1,
                                                                                  average_perplexity)
                    path = saver.save(session, os.path.join(checkpoint_dir, model_name))

                    print("Saved model checkpoint to {}".format(path))

                    if best_perplexity is None or best_perplexity > average_perplexity:
                        best_perplexity = average_perplexity
                        best_model = model_name

                print('Done with epoch', epoch + 1)
            
            best_model = "modelmodel_exp-B_epoch-4_val-perplexity-67.52006851045576"

            if best_model is None:
                raise Exception("Model has not been saved. Run for at least one epoch")

            print('Restoring best model', best_model)
            saver.restore(session, os.path.join(checkpoint_dir, best_model))

            # evaluate on test set
            perplexities = dev_step(get_batches(data_processing.test_corpus, batch_size=FLAGS.batch_size,
                                                do_shuffle=False), lstm, global_step, session, verbose=0)

            print('Perplexity on test_set is', np.mean(perplexities))

            filename = "perplexities-exp-{}.out".format(experiment_type)
            savefile = os.path.join(save_results_folder, filename)
            print('Saving results to', savefile)

            with open(savefile, 'w') as f:
                f.writelines(str(i) + '\n' for i in perplexities)

            if experiment_type == 'C':
                continuation_sentences = continue_sentences(data_processing.continuation_corpus, session,
                                                            lstm, data_processing)

                filename = "continuation-sentences-exp-{}.out".format(experiment_type)
                savefile = os.path.join(save_results_folder, filename)
                print('Saving results to', savefile)

                with open(savefile, 'w') as f:
                    f.writelines(str(i) + '\n' for i in continuation_sentences)

    print('Done')


if __name__ == '__main__':

    # needed to avoid errors
    tf.flags.DEFINE_string('f', '', 'kernel')

    tf.flags.DEFINE_integer("embedding_size", 100, "embedding size (default 100)")
    tf.flags.DEFINE_integer("batch_size", 64, "batch Size (default: 64)")
    tf.flags.DEFINE_integer("max_vocabulary_size", 20000, "Maximum vocabulary size (default: 20000)")
    tf.flags.DEFINE_integer("num_epochs", 5, "Number of training epochs (default: 10)")
    tf.flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many epochs (default: 1)")
    tf.flags.DEFINE_integer("num_checkpoints", 10, "Number of checkpoints to store (default: 10)")
    tf.flags.DEFINE_integer("sentence_length", 30, "Maximum length of a sentence (default 30)")
    tf.flags.DEFINE_string("experiment", "A", "Experiment to run (default A)")
    tf.flags.DEFINE_string("data_folder", None, "Data folder (default None)")
    tf.flags.DEFINE_string("save_model_folder", None, "Folder to save the models (default None)")
    tf.flags.DEFINE_string("save_results_folder", None, "Folder to save results (default None)")

    FLAGS = tf.app.flags.FLAGS

    if FLAGS.data_folder is None or FLAGS.save_model_folder is None or FLAGS.save_results_folder is None:
        raise SystemExit("Not all paths specified")
    
    print(" ")
    print('Running experiment type:', FLAGS.experiment)
    
    if not os.path.exists(FLAGS.data_folder):
        raise OSError("Data directory", FLAGS.data_folder, 'not found')

    if not os.path.exists(FLAGS.save_model_folder):
        os.makedirs(FLAGS.save_model_folder)

    if not os.path.exists(FLAGS.save_results_folder):
        os.makedirs(FLAGS.save_results_folder)

    run_experiment(FLAGS.experiment, FLAGS.data_folder, FLAGS.save_model_folder, FLAGS.save_results_folder)
