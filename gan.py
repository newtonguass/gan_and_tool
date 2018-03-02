import numpy as np
import tensorflow as tf

class ganpy:
    def __init__(self):
        self.epsilon = tf.constant(1e-8, tf.float32)
        self.on_train = tf.placeholder(dtype=tf.bool, shape=(), name='on_train')
        self.moving_average_rate = 0.99999
    def activation_function(self, image, function_type="elu", 
                            name="activation", lrelu_rate=0.2):
        if function_type == "elu":
            return tf.nn.elu(image, name=function_type)
        elif function_type == "relu":
            return tf.nn.relu(image, name=function_type)
        elif function_type == "lrelu":
            return tf.maximum(image, lrelu_rate*image, name=function_type)
        elif function_type == "relu6":
            return tf.nn.relu6(image, name=function_type)
        elif function_type == "tanh":
            return tf.nn.tanh(image, name=function_type)
        elif function_type == "sigmoid":
            return tf.nn.sigmoid(image, name =function_type)
        elif function_type == "selu":
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946             
            return scale * tf.where(image >= 0.0, image, alpha * tf.nn.elu(image), name =function_type )
        elif function_type == "swish":
            return image*tf.nn.sigmoid(image)
        elif function_type == "relu_1":
            return tf.minimum(tf.maximum(image, -1), 1)
        elif function_type == "same":
            return image
    
    def batch_norm(self, image, name="batch_norm"):
        print("batch_norm")
        axis_len = len(image.get_shape().as_list())
        axis = [i for i in range(axis_len-1)]
        channers_num = image.get_shape().as_list()[-1]
        
        with tf.variable_scope(name):
            mean, variance = tf.nn.moments(image, axes=axis)
            
            scale = tf.get_variable(name = 'alpha', 
                                    dtype=tf.float32, 
                                    shape=[channers_num],
                                    initializer=tf.ones_initializer())
            offset = tf.get_variable(name = 'beta', 
                                     dtype=tf.float32, 
                                     shape=[channers_num],
                                     initializer=tf.zeros_initializer())
            return tf.nn.batch_normalization(image, mean, variance, 
                                             offset, scale, self.epsilon,
                                            name="batch_norm_output")
        
        
    def linear_project(self, image, output_size, bn=False, 
                       function_type="elu",name='linear_project'):
        temp = image.get_shape().as_list()
        length = 1
        for i in range(1,len(temp)):
            length *= temp[i]
        image = tf.reshape(image, [-1, length])
        input_len = image.get_shape().as_list()[1]
        output_flatten_length = 1
        for i in output_size:
            output_flatten_length *= i
        with tf.variable_scope(name):
            initializer = tf.random_normal_initializer(dtype=tf.float32, 
                                                       stddev=1/input_len)
            weight = tf.get_variable(name='weight', 
                                     dtype=tf.float32, 
                                     shape=[input_len, 
                                            output_flatten_length],
                                    initializer = initializer)
            image = tf.matmul(image, weight, name="mutiply")
            if bn:
                image = self.batch_norm(image, name='bn')
            else:
                bias = tf.get_variable(name='bias', 
                                       dtype=tf.float32, 
                                       shape=output_flatten_length, 
                                       initializer=tf.zeros_initializer())
                tf.add(image, bias, name='add_bias')
            image = self.activation_function(image,
                                             function_type=function_type,
                                             name="avtivation_function")
            image = tf.reshape(image, [-1]+output_size)
            return image

    def conv2d(self, image, output_dim, kernel_size=3, strides=2, 
               bn=True, function_type="elu" ,name="con2d"):
        strides = [1, strides, strides, 1]
        with tf.variable_scope(name):
            input_channel = image.get_shape().as_list()[-1]
            initializer = tf.random_normal_initializer(dtype=tf.float32,
                                                        stddev=1/((kernel_size**2)*input_channel))
            kernel = tf.get_variable(name='kernel',
                                     dtype=tf.float32, 
                                     shape =[kernel_size,
                                             kernel_size,
                                             input_channel,
                                             output_dim],
                                     initializer = initializer)
        
                
            image = tf.nn.conv2d(image,
                                  kernel,
                                  padding='SAME',
                                  strides = strides,
                                  name="tf.nn.conv2d")
            if bn:
                image = self.batch_norm(image, name='bn')
            else:
                bias = tf.get_variable(name='bias', 
                                       dtype=tf.float32, 
                                       shape=[output_dim],
                                       initializer=tf.zeros_initializer())
                image = tf.add(image, bias, name="add_bias")
            image = self.activation_function(image,
                                             function_type=function_type,
                                             name="avtivation_function")
            return image
    
    def encoder(self, image, encoder_size , function_type="elu",
                kernel_size=3 ,bn=False, conv_loop=1 ,name='encoder', drop=None):
        
        
        with tf.variable_scope(name):    
            for i in range(len(encoder_size)):
                
                image = self.conv2d(image, 
                                   encoder_size[i][-1],
                                   kernel_size=kernel_size, 
                                   strides=2, 
                                   bn=bn, 
                                   function_type = function_type,
                                   name=name+"_layer_"+str(i)+"_0")
                if conv_loop>1:
                    by_pass =  tf.identity(image, name="by_pass"+str(i))
                print(image)
                for j in range(conv_loop-1):                        
                    image = self.conv2d(image, 
                                       encoder_size[i][-1],
                                       kernel_size=kernel_size, 
                                       strides=1, 
                                       bn=bn, 
                                       function_type = function_type,
                                       name=name+"_layer_"+str(i)+"_"+str(j+1))
                    print(image)
                if conv_loop>1:
                    image = image + by_pass
          
            #image = tf.reduce_mean(image, reduction_indices=[1, 2], name='avg_pool')
            print(image)
            
            return image
        
    def decoder(self, image, decoder_size, function_type="elu",
                kernel_size=3, bn=False, conv_loop=1,name='decoder', drop=None):
       
    
        with tf.variable_scope(name):
            
            print(image)
            #image = self.linear_project(image, decoder_size[0], name='linear_project')
            for i in range(1, len(decoder_size)):
            
                image = tf.image.resize_images(image, 
                                               decoder_size[i][0:2],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                if i==len(decoder_size)-1 :
                        function_type = "tanh"
                        bn=False
                for j in range(conv_loop):
                    
                    image = self.conv2d(image, 
                                        decoder_size[i][-1],
                                        kernel_size=kernel_size, 
                                        strides=1, 
                                        bn=bn, 
                                        function_type = function_type,
                                        name=name+"_layer_"+str(i)+"_"+str(j))
                    print(image)
                    if conv_loop>1 and j==0:
                        by_pass =  tf.identity(image, name="by_pass"+str(i))
                if conv_loop>1 :
                    image = image + by_pass
            
            print(image)
            
            
            return image 
    
    def discriminator(self, image, encoder_size , function_type="elu",
                kernel_size=3, strides=2,bn=True, conv_loop=1 ,name='discriminator', drop=None):
        
        with tf.variable_scope(name):    
            for i in range(len(encoder_size)-1):
                if drop!=None:                
                    image = tf.nn.dropout(image, keep_prob=drop)
                image = self.conv2d(image, 
                                   encoder_size[i][-1],
                                   kernel_size=kernel_size, 
                                   strides=2, 
                                   bn=bn, 
                                   function_type = function_type,
                                   name=name+"_layer_"+str(i)+"_0")
                if conv_loop>1:
                    by_pass =  tf.identity(image, name="by_pass"+str(i))
                print(image)
                for j in range(conv_loop-1):                        
                    image = self.conv2d(image, 
                                       encoder_size[i][-1],
                                       kernel_size=kernel_size, 
                                       strides=1, 
                                       bn=bn, 
                                       function_type = function_type,
                                       name=name+"_layer_"+str(i)+"_"+str(j+1))
                    print(image)
                if conv_loop>1:
                    image = image + by_pass
          
            image = tf.reduce_mean(image, reduction_indices=[1, 2], name='avg_pool')
            print(image)
            image = self.linear_project(image, 
                                        [1], 
                                        bn=False,
                                        function_type = "relu_1",
                                        name='linear_project')
            print(image)
            return image
   