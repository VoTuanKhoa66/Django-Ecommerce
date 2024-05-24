import datetime
from django.shortcuts import render, get_object_or_404
from .cart import Cart
from store.models import Product
from django.http import JsonResponse
from django.contrib import messages
# Create your views here.
def cart(request):
    cart = Cart(request)
    cart_products = cart.get_prods
    quantities = cart.get_quants
    totals = cart.total()
    date = datetime.datetime.now() + datetime.timedelta(days=3)
    formatted_date = date.strftime("%d-%m-%Y")
    return render(request, "cart.html", {"cart_products" : cart_products, "quantities":quantities, "totals":totals, "date":formatted_date})

def cart_add(request):
    cart = Cart(request)
    if request.POST.get('action') == 'post':
        product_id = int(request.POST.get('product_id'))
        product_qty = int(request.POST.get('product_qty'))
        #lookup product in DB
        product = get_object_or_404(Product, id=product_id)
        #save to session
        cart.add(product=product, quantity=product_qty)

        cart_quantity = cart.__len__()

        # response = JsonResponse({'Product Name:': product.name})
        response = JsonResponse({'qty': cart_quantity})
        messages.success(request, ("Item added to cart!"))
        return response

def cart_update(request):
    cart = Cart(request)
    if request.POST.get('action') == 'post':
        product_id = int(request.POST.get( 'product_id' ))
        product_qty = int(request.POST.get( 'product_qty' ))
        
        cart.update(product=product_id, quantity=product_qty )
        response = JsonResponse({'qty': product_qty})
        messages.success(request, ("Item was updated."))
        return response
        # return redirect('cart')

def cart_delete(request):
    cart = Cart(request)
    if request.POST.get('action') == 'post':
        product_id = int(request.POST.get( 'product_id' ))
        cart.delete(product=product_id)
        
        response = JsonResponse({'product': product_id})
        messages.success(request, ("1 item removed"))
        return response