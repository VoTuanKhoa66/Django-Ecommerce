import json
import ssl
from django.shortcuts import render, redirect

from .models import Product, Category, Profile
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.models import User
from .forms import SignUpForm, UpdateUserForm, ChangePasswordForm, UserInfoForm
from payment.forms import ShippingForm
from payment.models import ShippingAddress
from cart.cart import Cart

import urllib.request
import pickle
import numpy as np
from numpy.linalg import norm
import urllib.request
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.layers import GlobalMaxPooling2D # type: ignore
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from sklearn.neighbors import NearestNeighbors
# Create your views here.


feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Create an unverified SSL context
ssl_context = ssl._create_unverified_context()

# Function to load and preprocess image from URL
def load_and_preprocess_image(url):
    # Download image from URL
    with urllib.request.urlopen(url,context=ssl_context) as response:
        img = np.asarray(bytearray(response.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)  # Convert bytes data to image array
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (224, 224))  # Resize image to match model input size
    img_array = np.expand_dims(img, axis=0)  # Add batch dimension
    preprocessed_img = preprocess_input(img_array)  # Preprocess input for ResNet50
    return preprocessed_img

def recommend(feature_list, url):
    preprocessed_img = load_and_preprocess_image(url)

    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    
    # Function to compute Euclidean distance
    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # Implementing KNN from scratch
    def knn(feature_list, query_feature, k=4):
        distances = []
        skipped_one = False  # Flag to keep track if one similar URL has been skipped
        for idx, feature in enumerate(feature_list):
            # Check if the filename matches the URL and if we haven't skipped one yet
            if filenames[idx] == url and not skipped_one:
                skipped_one = True
                continue  # Skip this one and move to the next iteration
            dist = euclidean_distance(feature, query_feature)
            distances.append((dist, idx))
    
        # Sort by distance
        distances.sort(key=lambda x: x[0])
        
        # Get the indices of the k nearest neighbors
        neighbors_indices = [idx for dist, idx in distances[:k]]
        
        return neighbors_indices
    nearest_neighbors = knn(feature_list, normalized_result, k=4)
    nearest_neighbors_urls = [filenames[idx] for idx in nearest_neighbors]

    return nearest_neighbors_urls

def search(request):
    if request.method == "POST":
        searched = request.POST['searched']
        result = Product.objects.filter(name__icontains=searched)
        
        if not result:
            searched = request.POST['searched']
            return render(request, 'search.html', {'searched':searched})
        else:
            return render(request, "search.html", {'searched':searched , 'result':result})
    else:
        return render(request, 'search.html', {})

def update_info(request):
    if request.user.is_authenticated:
        current_user = Profile.objects.get(user__id=request.user.id)
        shipping_user = ShippingAddress.objects.get(user__id=request.user.id)
        form = UserInfoForm(request.POST or None, instance=current_user)
        shipping_form = ShippingForm(request.POST or None, instance=shipping_user)
        if form.is_valid() or shipping_form.is_valid():

            form.save()
            shipping_form.save()

            messages.success(request, "Your Info updated successfully!")
            return redirect('update_info')
        return render(request, 'update_info.html', {'form': form, 'shipping_form':shipping_form})
    else:
        messages.success(request, "You must  be logged in to view this page.")
        return redirect('home')


def update_password(request):
    if request.user.is_authenticated:
        current_user = request.user
        if request.method == 'POST':  
            form = ChangePasswordForm(current_user, request.POST)
            if form.is_valid():
                form.save()
                messages.success(request, "Your password has been updated successfully.") 
                login(request, current_user) # re-login the user after successful password change
                return redirect('update_profile')
            else:
                for error in list(form.errors.values()):
                    messages.error(request, error)
                    return redirect('update_password')
        else:
            form = ChangePasswordForm(current_user)
            return render(request, 'update_password.html', {'form' : form})
    else:
        messages.success(request, "You must  be logged in to view this page.")
        return redirect('home')

def update_profile(request):
    if request.user.is_authenticated:
        current_user = User.objects.get(id = request.user.id)
        user_form = UpdateUserForm(request.POST or None, instance=current_user)
        if user_form.is_valid():
            user_form.save()
            login(request, current_user)
            messages.success(request, "Profile updated successfully!")
            return redirect('update_profile')
        return render(request, 'update_profile.html', {'user_form': user_form})
    else:
        messages.success(request, "You must  be logged in to view this page.")
        return redirect('home')

def category(request, foo):
    foo = foo.replace('-',' ')
    try:
        category = Category.objects.get(name=foo)
        products = Product.objects.filter(category=category)
        return render(request, 'category.html',{'products':products, 'category':category})
    except:
        messages.success(request, ("That Category Doesn't Exit !"))
        return redirect('home')
    

def product(request,pk):
    product = Product.objects.get(id=pk)
    image = product.image_url
    product_recommend = recommend(feature_list, image)
    
    # Lấy các sản phẩm theo URL
    recommended_products = Product.objects.filter(image_url__in=product_recommend)

    # In danh sách các id ra terminal
    print("Recommended product :", recommended_products)

    return render(request, 'product.html', {'product': product, 'recommend': recommended_products})

def home(request): 
    products = Product.objects.all()
    return render(request, 'home.html', {'products': products})

def about(request):
    return render(request, 'about.html', {})

def login_user(request):
    
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if  user is not None:
            login(request, user)

            current_user = Profile.objects.get(user__id = request.user.id)

            saved_cart = current_user.old_cart

            #covert
            if saved_cart:
                coverted_cart = json.loads(saved_cart)
                cart = Cart(request)
                for key, value in  coverted_cart.items():
                    cart.db_add(product=key, quantity=value)
            # Redirect to a success page.
            messages.success(request, ("You are now logged in."))
            return redirect('home')
        else:
            messages.success(request, ("There was an error!"))
            return redirect('login')
    else:
        return render(request, 'login.html', {})

def logout_user(request):
    logout(request)
    messages.success(request, ("You have been logged out."))
    return redirect('home')

def register_user(request):
    form = SignUpForm()
    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data['username']
            password = form.cleaned_data['password1']
            user = authenticate(username=username, password=password)
            login(request, user)
            messages.success(request, ("Sign Up Success!! - Please fill in your information"))
            return redirect('update_info')
        else:
            messages.success(request, ("There was a problem Regitering, please try again!"))
            return redirect('register')
    else:      
        return render(request, 'register.html',{'form':form})