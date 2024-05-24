from django.contrib import messages
from django.shortcuts import redirect, render
from cart.cart import Cart
from payment.forms import ShippingForm, PaymentForm
from payment.models import ShippingAddress
import stripe
from django.conf import settings
# Create your views here.
stripe.api_key = settings.STRIPE_SECRET_KEY

def payment_cash(request):
    cart = Cart(request)
    cart.deleteAll()
    messages.success(request, 'Your order has been processed!')
    return redirect('home')


def process_payment(request):
    if request.method == 'POST':
        cart = Cart(request)
        totals = cart.total()
        try:
            customer = stripe.Customer.create(
                email=request.user.email,
                name=request.user.first_name,
            )
            customer_id = customer.id

            card = stripe.Customer.create_source(
                customer=customer_id,
                source='tok_visa'
            )
            # Tạo một giao dịch thanh toán mới trong Stripe
            charge = stripe.Charge.create(
                amount=int(100 * totals),
                currency='usd',
                customer=customer_id,
                source=card.id
            )
            if charge:
                cart.deleteAll()
            # charge_id = charge.id
            # print("Charge ID:", charge_id)
            messages.success(request, 'Payment processed successfully!')
            return redirect('home')
        except stripe.error.CardError as e:
            # Xử lý lỗi thẻ bị từ chối
            messages.success(request, 'Your card was declined.')
            return redirect('payment_card')
    else:
        return render(request, 'payment/checkout.html')



def checkout(request):

    cart = Cart(request)
    cart_products = cart.get_prods
    quantities = cart.get_quants
    totals = cart.total()
    if request.user.is_authenticated:
		# Checkout as logged in user
		# Shipping User
        shipping_user = ShippingAddress.objects.get(user__id=request.user.id)
		# Shipping Form
        shipping_form = ShippingForm(request.POST or None, instance=shipping_user)
        return render(request, "payment/checkout.html", {"cart_products":cart_products, "quantities":quantities, "totals":totals, "shipping_form":shipping_form })
    else:
		# Checkout as guest
        # shipping_form = ShippingForm(request.POST or None)

        # return render(request, "payment/checkout.html", {"cart_products":cart_products, "quantities":quantities, "totals":totals, "shipping_form":shipping_form})
        messages.success(request, 'Please login to complete the order')
        return redirect('login')
    
def billing(request):
    if request.POST:
        cart = Cart(request)
        cart_products = cart.get_prods
        quantities = cart.get_quants
        totals = cart.total()

        if request.user.is_authenticated:
            return render(request, "payment/billing.html", {"cart_products":cart_products, "quantities":quantities, "totals":totals, "shipping_info":request.POST })
        else:
            pass

        shipping_form = request.POST
        return render(request, "payment/billing.html", {"cart_products":cart_products, "quantities":quantities, "totals":totals, "shipping_form":shipping_form })
    else:
        messages.success(request, 'Access Denied')
        return redirect('home')