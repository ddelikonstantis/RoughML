# takes a model and prints out certain information
# layer name, layer type, layer weights shape, layer weights, layer requires_grad


def model_weights(model):
    # Display all model layer weights
    for name, param in model.named_parameters():
        print('name: ', name)
        print(type(param))
        print('param.shape: ', param.shape)
        print('param.shape: ', param)
        print('param.requires_grad: ', param.requires_grad)
        print('=====')