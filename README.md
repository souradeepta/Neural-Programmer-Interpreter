# Program Induction using Neural Programmer-Interpreter

This project is based on the [Neural Programmer-Interpreter](https://arxiv.org/abs/1511.06279) paper, by Reed and de Freitas.

## Introduction ##

The Neural Programmer-Interpreter is made up of below components:

+ **State Encoder**: This network generates an encoding of the inputs created by merging the environment and previous subroutine arguments. It generates a fixed-length state encoding. (**f_enc**)
  
- This encoder is task-specific and must be designed based on task environment
  
+ **Core**: A 2 layer LSTM network trained to encode temporal information of which subroutine will be called in the next timestemp and with what arguments. (**f_lstm**)
    - Inputs are the encoded state and the required program embedding (**m_prog**)
    - This component is common for all tasks.

+ **Program End Predictor**: Evaluates the hidden state from the LSTM layer and predicts the probability of terminating the execution (**f_end**)
    - This is a Feed-Forward network
    - Based on paper applies a threshold of 0.5 to decide on program termination

+ **Next Program Predictor**: Evaluates the program key embedding which is used to compile the next program to be executed (**f_prog**)
  
- The probabilities for the next program are generated using a softmax function.
  
+ **Arguments Predictor**: Evaluates the program arguments based on the hidden state from the LSTM layer (**f_arg**)
    - This is a Feed-Forward network

        
### Project Setup ###
    + model/
        - Contains code for the NPI Core, Program End Predictor, Next Program Predictor and Arguments Predictor
        
    + tasks/
        - Has a folder for each task implemented
        - It contains modules to generate traces, train and evaluate the task
    
    + main.py 
        - The executable for selected task.


### Executing this Project ###

**main.py** has below configuration flags:

```
    tf.app.flags.DEFINE_string("task", "card_pattern_matching", "Which NPI Task to run - [addition, card_pattern_matching, merge_sort].")

    tf.app.flags.DEFINE_boolean("generate", True, "Boolean whether to generate training/test data.")
    tf.app.flags.DEFINE_integer("num_training", 1000, "Number of training examples to generate.")
    tf.app.flags.DEFINE_integer("num_test", 100, "Number of test examples to generate.")

    tf.app.flags.DEFINE_boolean("do_train", False, "Boolean whether to continue training model.")
    tf.app.flags.DEFINE_boolean("do_eval", False, "Boolean whether to perform model evaluation.")
    tf.app.flags.DEFINE_integer("num_epochs", 2, "Number of training epochs to perform.")
```

By default the application is setup to generate the traces for card_pattern_matching task. This can be modified by changing the ```task`` attribute.

The ```generate``` flag is the only one turned-on by default. The ```do_train``` and ```do_eval``` flags can also be turned-on to execute those functions.

```
    ## Example 1 ##
    ### Card 1: 2 of spades , Card 2: 2 of spades ###

    Card 1 :  022
    Card 2 :  022
    -------------
    Output :  000

    Card 1 :  022
    Card 2 :  022
    -------------
    Output :  000

    Card 1 :  022
    Card 2 :  022
    -------------
    Output :  000

    {
        'card1': {'suit': 'spades', 'rank': '2'}, 
        'card2': {'suit': 'spades', 'rank': '2'}, 
        'traces': [
            (('CMP', 2), [], False), 
            (('USUB1', 3), [], False), 
            (('WRITE', 1), [0, 0], False), 
            (('MOVE_PTR', 0), [0, 0], False), 
            (('MOVE_PTR', 0), [1, 0], False), 
            (('MOVE_PTR', 0), [2, 0], False), 
            (('USUB1', 3), [], False), 
            (('WRITE', 1), [0, 0], False), 
            (('MOVE_PTR', 0), [0, 0], False), 
            (('MOVE_PTR', 0), [1, 0], False), 
            (('MOVE_PTR', 0), [2, 0], False), 
            (('USUB1', 3), [], False), 
            (('WRITE', 1), [0, 0], False), 
            (('MOVE_PTR', 0), [0, 0], False), 
            (('MOVE_PTR', 0), [1, 0], False), 
            (('MOVE_PTR', 0), [2, 0], True)
        ]
    }

    ## Example 2 ##
    ### Card 1: 7 of diamonds , Card 2: 6 of diamonds ###

    Card 1 :  074
    Card 2 :  064
    -------------
    Output :  000

    Card 1 :  074
    Card 2 :  064
    -------------
    Output :  010

    {
        'card1': {'suit': 'diamonds', 'rank': '7'}, 
        'card2': {'suit': 'diamonds', 'rank': '6'}, 
        'traces': [
            (('CMP', 2), [], False), 
            (('USUB1', 3), [], False), 
            (('WRITE', 1), [0, 0], False), 
            (('MOVE_PTR', 0), [0, 0], False), 
            (('MOVE_PTR', 0), [1, 0], False), 
            (('MOVE_PTR', 0), [2, 0], False), 
            (('USUB1', 3), [], False), 
            (('WRITE', 1), [0, 1], False), 
            (('MOVE_PTR', 0), [0, 0], False), 
            (('MOVE_PTR', 0), [1, 0], False), 
            (('MOVE_PTR', 0), [2, 0], True)
        ]
    }

    ## Example 3 ##
    ### Card 1: 9 of hearts , Card 2: A of clubs ###

    Card 1 :  091
    Card 2 :  013
    -------------
    Output :  002

    {
        'card1': {'suit': 'hearts', 'rank': '9'}, 
        'card2': {'suit': 'clubs', 'rank': 'A'}, 
        'traces': [
            (('CMP', 2), [], False), 
            (('USUB1', 3), [], False), 
            (('WRITE', 1), [0, 2], False), 
            (('MOVE_PTR', 0), [0, 0], False), 
            (('MOVE_PTR', 0), [1, 0], False), 
            (('MOVE_PTR', 0), [2, 0], True)
        ]
    }

```