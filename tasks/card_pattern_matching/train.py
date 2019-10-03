import pickle
import tensorflow as tf

from model.npi import NPI

from tasks.card_pattern_matching.card_pattern_matching import CardPatternMatchingCore
from tasks.card_pattern_matching.config import CONFIG
from tasks.card_pattern_matching.env import ScratchPad, get_args


MOVE_PID, WRITE_PID = 0, 1
DATA_PATH = "tasks/card_pattern_matching/data/card_pattern_matching_train.pik"
LOG_PATH = "tasks/card_pattern_matching/log/"

def train_card_pattern_matching(epochs, verbose=0):
    # Load Data
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    # Initialize Card Pattern Matching Core
    print ('Initializing Card Pattern Matching Core!')
    core = CardPatternMatchingCore()

    # Initialize NPI Model
    print ('Initializing NPI Model!')
    npi = NPI(core, CONFIG, LOG_PATH, verbose=verbose)

    # Initialize TF Saver
    saver = tf.train.Saver()

    # Initialize TF Session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Start Training
        for ep in range(1, epochs + 1):
            for i in range(len(data)):
                # Reset NPI States
                npi.reset_state()

                # Setup Environment
                card1, card2, steps = data[i]
                scratch = ScratchPad(card1, card2)
                x, y = steps[:-1], steps[1:]

                # Run through steps, and fit!
                step_def_loss, step_arg_loss, term_acc, prog_acc, = 0.0, 0.0, 0.0, 0.0
                arg0_acc, arg1_acc, arg2_acc, num_args = 0.0, 0.0, 0.0, 0
                for j in range(len(x)):
                    (prog_name, prog_in_id), arg, term = x[j]
                    (_, prog_out_id), arg_out, term_out = y[j]

                    # Update Environment if MOVE or WRITE
                    if prog_in_id == MOVE_PID or prog_in_id == WRITE_PID:
                        scratch.execute(prog_in_id, arg)

                    # Get Environment, Argument Vectors
                    env_in = [scratch.get_env()]
                    arg_in, arg_out = [get_args(arg, arg_in=True)], get_args(arg_out, arg_in=False)
                    prog_in, prog_out = [[prog_in_id]], [prog_out_id]
                    term_out = [1] if term_out else [0]

                    # Fit!
                    if prog_out_id == MOVE_PID or prog_out_id == WRITE_PID:
                        loss, t_acc, p_acc, a_acc, _ = sess.run(
                            [npi.arg_loss, npi.t_metric, npi.p_metric, npi.a_metrics, npi.arg_train_op],
                            feed_dict={npi.env_in: env_in, npi.arg_in: arg_in, npi.prg_in: prog_in,
                                    npi.y_prog: prog_out, npi.y_term: term_out,
                                    npi.y_args[0]: [arg_out[0]], npi.y_args[1]: [arg_out[1]],
                                    npi.y_args[2]: [arg_out[2]]})
                        step_arg_loss += loss
                        term_acc += t_acc
                        prog_acc += p_acc
                        arg0_acc += a_acc[0]
                        arg1_acc += a_acc[1]
                        arg2_acc += a_acc[2]
                        num_args += 1
                    else:
                        loss, t_acc, p_acc, _ = sess.run(
                            [npi.default_loss, npi.t_metric, npi.p_metric, npi.default_train_op],
                            feed_dict={npi.env_in: env_in, npi.arg_in: arg_in, npi.prg_in: prog_in,
                                    npi.y_prog: prog_out, npi.y_term: term_out})
                        step_def_loss += loss
                        term_acc += t_acc
                        prog_acc += p_acc

                print ("Epoch {0:02d} Step {1:03d} Default Step Loss {2:05f}, " \
                    "Argument Step Loss {3:05f}, Term: {4:03f}, Prog: {5:03f}, A0: {6:03f}, " \
                    "A1: {7:03f}, A2: {8:03}".format(ep, i, step_def_loss / len(x), step_arg_loss / len(x), term_acc / len(x),
                            prog_acc / len(x), arg0_acc / num_args, arg1_acc / num_args,
                            arg2_acc / num_args))

            # Save Model
            saver.save(sess, 'tasks/card_pattern_matching/log/card_pattern_matching_model.ckpt')

    print ('Model generation complete!')

if __name__ == '__main__':
    train_card_pattern_matching()

