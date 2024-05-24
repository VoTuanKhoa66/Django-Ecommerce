from store.models import Product, Profile

class Cart():
    def __init__(self, request):
        self.session = request.session
        self.request = request
        #Get the current session key if it exists
        cart = self.session.get('session_key')

        #if the user is new, no session key
        if 'session_key' not in request.session:
            cart = self.session['session_key'] = {}

        #make sure cart is available on all pages    
        self.cart = cart

    def db_add(self, product, quantity):
        product_id = str(product)
        product_qty = str(quantity)

        if product_id in self.cart:
            pass
        else:
            # self.cart[product_id] = {'price':str(product.price)}
            self.cart[product_id] = int(product_qty)

        self.session.modified = True

        #Deal with logged in user
        if self.request.user.is_authenticated:
            current_user = Profile.objects.filter(user__id = self.request.user.id)
            # Covert
            carty = str(self.cart)
            carty = carty.replace("\'", "\"")

            #save carty to the profile model
            current_user.update(old_cart=str(carty))

    def add(self, product, quantity):
        product_id = str(product.id)
        product_qty = str(quantity)

        if product_id in self.cart:
            pass
        else:
            # self.cart[product_id] = {'price':str(product.price)}
            self.cart[product_id] = int(product_qty)

        self.session.modified = True

        #Deal with logged in user
        if self.request.user.is_authenticated:
            current_user = Profile.objects.filter(user__id = self.request.user.id)
            # Covert
            carty = str(self.cart)
            carty = carty.replace("\'", "\"")

            #save carty to the profile model
            current_user.update(old_cart=str(carty))
            
    def total(self):
        quantities = self.cart
        product_ids = self.cart.keys()
        products = Product.objects.filter(id__in=product_ids)

        total = 0
        for key, value in quantities.items():
            key = int(key)
            for product in products:
                if product.id == key:
                    if product.is_sale:
                        total = total + (product.sale_price * value)
                    else:
                        total = total + (product.price * value)
        return total

    def __len__(self):
        return len(self.cart)
    
    def get_prods(self):
        product_ids = self.cart.keys()
        products = Product.objects.filter(id__in = product_ids)

        return products
    
    def get_quants(self):
        quantities = self.cart
        return quantities
    
    def update(self, product, quantity):
        product_id = str(product)
        product_qty = int(quantity)

        ourcart = self.cart

        ourcart[product_id] = product_qty

        self.session.modified = True

        #Deal with logged in user
        if self.request.user.is_authenticated:
            current_user = Profile.objects.filter(user__id = self.request.user.id)
            # Covert
            carty = str(self.cart)
            carty = carty.replace("\'", "\"")

            #save carty to the profile model
            current_user.update(old_cart=str(carty))

        thing = self.cart
        
        return thing
    
    def delete(self, product):
        product_id = str(product)
        if  product_id in self.cart:
            del self.cart[product_id]

        self.session.modified = True

        #Deal with logged in user
        if self.request.user.is_authenticated:
            current_user = Profile.objects.filter(user__id = self.request.user.id)
            # Covert
            carty = str(self.cart)
            carty = carty.replace("\'", "\"")

            #save carty to the profile model
            current_user.update(old_cart=str(carty))

    def deleteAll(self):
        # Gán old_cart bằng một dictionary rỗng
        self.cart = {}
        self.session['session_key'] = self.cart
        self.session.modified = True

        # Deal with logged in user
        if self.request.user.is_authenticated:
            current_user = Profile.objects.filter(user__id=self.request.user.id)
            # Gán old_cart của user bằng một dictionary rỗng
            current_user.update(old_cart={})