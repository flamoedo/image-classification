import tensorflow as tf
import problem_unittests as tests
import helper


def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    
    weights = tf.Variable(tf.random_normal([int(x_tensor.shape[1]), num_outputs]))
    
    biases = tf.Variable(tf.random_normal([num_outputs]))
    
    fc1 = tf.add(tf.matmul(x_tensor, weights), biases)
    
    return fc1


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_fully_conn(fully_conn)