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
        if len(temp)-1 == len(output_size):
            reshape = False
        else:
            reshape = True

        length = 1
        for i in range(1,len(temp)):
            length *= temp[i]
        if reshape:
            image = tf.reshape(image, [-1, length])
        input_len = image.get_shape().as_list()[1]
        output_flatten_length = 1
        for i in output_size:
            output_flatten_length *= i
        with tf.variable_scope(name):
            initializer = tf.random_normal_initializer(dtype=tf.float32, 
                                                       stddev=2/input_len)
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
            if reshape:
                image = tf.reshape(image, [-1]+output_size)
            return image

    def conv2d(self, image, output_dim, kernel_size=3, strides=2, 
               bn=False, function_type="elu" ,name="con2d"):
        strides = [1, strides, strides, 1]
        with tf.variable_scope(name):
            input_channel = image.get_shape().as_list()[-1]
            initializer = tf.random_normal_initializer(dtype=tf.float32,
                                                        stddev=2/((kernel_size**2)*input_channel))
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
    
    def residual_block(self, image, output_channel, function_type="elu",
                kernel_size=3 ,bn=False, down_sample=False, conv_bypass=False,residual_=2,
                name='residual_block'):
        if down_sample:
            stride_=2
        else:
            
            stride_=1
        if conv_bypass:
            by_pass = self.conv2d(image, 
                               output_channel,
                               kernel_size=kernel_size, 
                               strides=stride_, 
                               bn=bn, 
                               function_type = 'same',
                               name=name+"_by_pass")
        else:
            by_pass = tf.identity(image, name=name+'_by_pass')
        print(by_pass)
        
        for i in range(residual_ ):
            if i>0:
                stride_=1
            if i<residual_-1:
                output_channel_ = output_channel//2
                non_linear = function_type
            else:
                output_channel_ = output_channel
                non_linear = 'same'
             
            image = self.conv2d(image, 
                               output_channel_,
                               kernel_size=3, 
                               strides=stride_, 
                               bn=bn, 
                               function_type = non_linear,
                               name=name+"_residual_"+str(i))
            print(image)
        image = image + by_pass
        image = self.activation_function(image, function_type=function_type)
        print(image)
        return image

    def residual_mlp(self, image, size, function_type='elu', bn=False , mlp_bypass=True, residual_=2, name='resudual_mlp'):
        if mlp_bypass:
            bypass = self.linear_project(image, 
                                        size, 
                                        bn=False,
                                        function_type = 'same',
                                        name=name+'_bypass')
        else:
            bypass = tf.identity(image, name=name+'_bypass')
        print(bypass)
        for i in range(residual_):
            if i<residual_-1:
                non_linear = function_type
                hidden = [size[0]//2]
            else:
                non_linear = 'same'
                hidden = size
            image = self.linear_project(image, 
                                        hidden, 
                                        bn=bn,
                                        function_type = non_linear,
                                        name=name+'_residual_'+str(i))
            print(image)
        image = bypass + image
        image = self.activation_function(image, function_type=function_type)
        return image

    def mlp(self, image, mlp_size, function_type="elu", bn=False, residual_block=2, residual_=2, name="mlp"):
        with tf.variable_scope(name):
            for i in range(len(mlp_size)):
                if i==len(mlp_size)-1:
                    bn = False
                for j in range(residual_block):
                    if j>0:
                        mlp_bypass=False
                    else:
                        mlp_bypass=True
                    image = self.residual_mlp(image,
                                mlp_size[i], 
                                function_type=function_type,
                                bn=bn,
                                mlp_bypass=mlp_bypass, 
                                residual_ = residual_,
                                name='resudual_mlp_'+str(i)+'_'+str(j))
           
            return image

    def mlp_d(self, image, mlp_size, function_type="elu", bn=False, residual_block=2, residual_=2, name="mlp"):
        with tf.variable_scope(name):
            for i in range(len(mlp_size)):
                for j in range(residual_block):
                    if j>0:
                        mlp_bypass=False
                    else:
                        mlp_bypass=True
                    image = self.residual_mlp(image,
                                mlp_size[i], 
                                function_type=function_type,
                                bn=bn,
                                mlp_bypass=mlp_bypass, 
                                residual_ = residual_,
                                name='resudual_mlp_'+str(i)+'_'+str(j))
            return image

    def bottleneck(self, image, encoder_size , function_type="elu",
                kernel_size=3,bn=False, residual_block=2,residual_=2 ,name='encoder'):
        with tf.variable_scope(name):    
            for i in range(len(encoder_size)):

                for j in range(residual_block):    
                    if j>0:
                        conv_bypass = False
                    else:
                        conv_bypass = True
                    image = self.residual_block(image,
                                    encoder_size[i][-1], 
                                    function_type=function_type,
                                    kernel_size=kernel_size ,
                                    bn=bn, 
                                    down_sample=False, 
                                    conv_bypass=conv_bypass,
                                    residual_ = residual_,
                                    name='residual_block_layer'+str(i)+'_'+str(j))
            return image
    def encoder(self, image, encoder_size , function_type="elu",
                kernel_size=3,bn=False, residual_block=2,residual_=2 ,name='encoder'):
        with tf.variable_scope(name):    
            for i in range(len(encoder_size)):

                for j in range(residual_block):    
                    if j>0:
                        down_sample=False
                        conv_bypass = False
                    else:
                        down_sample=True
                        conv_bypass = True
                    image = self.residual_block(image,
                                    encoder_size[i][-1], 
                                    function_type=function_type,
                                    kernel_size=kernel_size ,
                                    bn=bn, 
                                    down_sample=down_sample, 
                                    conv_bypass=conv_bypass,
                                    residual_ = residual_,
                                    name='residual_block_layer'+str(i)+'_'+str(j))


            return image
        

    def decoder(self, image, decoder_size, function_type="elu",
                kernel_size=3, bn=False, residual_block=2, residual_=2,loss_layer=None, name='decoder'):
        with tf.variable_scope(name):
            outputs = [] 
            print(image)

            for i in range(len(decoder_size)):
            
                image = tf.image.resize_images(image, 
                                               decoder_size[i][0:2],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                for j in range(residual_block):
                    if j>0:
                        conv_bypass = False
                    else:
                        conv_bypass = True
                    image = self.residual_block(image,
                                    decoder_size[i][-1], 
                                    function_type=function_type,
                                    kernel_size=3 ,
                                    bn=bn, 
                                    down_sample=False,
                                    conv_bypass=conv_bypass,
                                    residual_ = residual_,
                                    name='residual_block_layer'+str(i)+'_'+str(j))
                if decoder_size[i][0]>=decoder_size[-loss_layer][0]:            
                    output = self.conv2d(image, 
                                        1,
                                        kernel_size=kernel_size, 
                                        strides=1, 
                                        function_type = 'tanh',
                                        name=name+"_output_"+str(i))
                    outputs.append(output)

            print(image)
            if loss_layer!=None:
                return outputs
            else:
                return image
            return image

    def discriminator(self, image, encoder_size , function_type="elu",
                kernel_size=3, strides=2,bn=True,name='discriminator'):
        
        with tf.variable_scope(name):    
            for i in range(len(encoder_size)):
                for j in range(2):    
                    if j>0:
                        down_sample=False
                        conv_bypass = False
                    else:
                        down_sample=True
                        conv_bypass = True
                    image = self.residual_block(image,
                                    encoder_size[i][-1], 
                                    function_type="elu",
                                    kernel_size=3 ,
                                    bn=bn, 
                                    down_sample=down_sample, 
                                    conv_bypass=conv_bypass,
                                    name='residual_block_layer'+str(i)+'_'+str(j))
            image = tf.reduce_mean(image, reduction_indices=[1, 2], name='avg_pool')
            print(image)
            image = self.linear_project(image, 
                                        [1], 
                                        bn=False,
                                        function_type = "relu_1",
                                        name='linear_project')
            print(image)
            return image
   
