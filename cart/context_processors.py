from .cart import Cart

def cart(request):
    #Return the default data from our cart
    return {'cart': Cart(request)}