
import random
import pickle

from tasks.card_pattern_matching.config import CARD_SUITS, CARD_RANKS
from tasks.card_pattern_matching.trace import Trace

def card(suit, rank):
    return {
        'suit': suit,
        'rank': rank
    }

def random_card(suit=None, rank=None):
    return card(
        suit=(random.choice(CARD_SUITS) if suit is None else suit),
        rank=(random.choice(CARD_RANKS) if rank is None else rank)
    )

def save_traces(output_tag, traces, debug=False):
    if debug:
        print(traces)
    with open('tasks/card_pattern_matching/data/card_pattern_matching_{}.pik'.format(output_tag), 'wb') as f:
        pickle.dump(traces, f)


def generate_random(suit=None, rank=None, num=1000, debug=False, debug_interval=50):
    traces = []

    for index in range(num):
        card1 = random_card(suit=suit, rank=rank)
        card2 = random_card(suit=suit, rank=rank)

        do_debug = (debug and index % debug_interval == 0)
        trace = Trace(card1, card2, debug=do_debug).trace

        traces.append(( card1, card2, trace ))

    return traces

def generate_same_rank(num=100, debug=False, debug_interval=50):
    traces = []
    for rank in CARD_RANKS:
        traces.extend(
            generate_random(
                rank=rank,
                num=num,
                debug=debug,
                debug_interval=debug_interval
            )
        )
    return traces

def generate_same_suit(num=100, debug=False, debug_interval=50):
    traces = []
    for suit in CARD_SUITS:
        traces.extend(
            generate_random(
                suit=suit,
                num=num,
                debug=debug,
                debug_interval=debug_interval
            )
        )
    return traces

def generate_matching_cards(debug=False, debug_interval=50):
    traces = []

    index = 0
    for suit in CARD_SUITS:
        for rank in CARD_RANKS:
            card1 = card2 = card(suit, rank)
            do_debug = (debug and index % debug_interval == 0)
            trace = Trace(card1, card2, debug=do_debug).trace
            traces.append(( card1, card2, trace ))
            index = index + 1

    return traces



def generate(output_tag, num=1000, only_random=False, debug=False, debug_interval=50):
    traces = []
    if not only_random:
        traces.extend(generate_matching_cards(debug=debug, debug_interval=debug_interval))
        traces.extend(generate_same_suit(debug=debug, debug_interval=debug_interval))
        traces.extend(generate_same_rank(debug=debug, debug_interval=debug_interval))

    traces.extend(generate_random(num=num, debug=debug, debug_interval=debug_interval))

    save_traces(output_tag, traces, debug)
    print('\n TOTAL TRACES GENERATED = ', len(traces), '\n')



if __name__ == '__main__':
    generate('train', debug=True)