from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.contrib import messages

from .forms import UserRegistrationForm, \
                    UserEditForm, ProfileEditForm
from .models import Profile
# Create your views here.


@login_required
def dashboard(request):

    user = request.user
    no_of_pc = user.perceptron_models_created.count()
    no_of_lr = user.logistic_regression_models_created.count()
    no_of_rg = user.regression_models_created.count()
    no_of_nn = user.neural_net_models_created.count()
    no_of_km = user.k_means_models_created.count()
    return render(request,
                'account/dashboard.html',
                {'section': 'dashboard',
                'user':user,
                'no_of_nn':no_of_nn,
                'no_of_pc':no_of_pc,
                'no_of_lr':no_of_lr,
                'no_of_rg':no_of_rg,
                'no_of_km':no_of_km,})



def register(request):
    if request.method == 'POST':
        user_form = UserRegistrationForm(request.POST)
        if user_form.is_valid():
            # Create a new user object but avoid saving it yet
            new_user = user_form.save(commit=False)
            # Set the chosen password
            new_user.set_password(
                user_form.cleaned_data['password'])
            # Save the User object
            new_user.save()
            # Create the user profile
            Profile.objects.create(user=new_user)
            return render(request,
                          'account/register_done.html',
                          {'new_user': new_user})
    else:
        user_form = UserRegistrationForm()
    return render(request,
                  'account/register.html',
                  {'user_form': user_form})


@login_required
def edit(request):
    if request.method == 'POST':
        user_form = UserEditForm(instance=request.user,
                                 data=request.POST)
        profile_form = ProfileEditForm(
                                    instance=request.user.profile,
                                    data=request.POST,
                                    files=request.FILES)
        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            messages.success(request, 'Profile updated successfully')
        else:
            messages.error(request, 'Error updating your profile')
    else:
        user_form = UserEditForm(instance=request.user)
        profile_form = ProfileEditForm(instance=request.user.profile)
    return render(request,
                  'account/edit.html',
                  {'user_form': user_form,
                   'profile_form': profile_form})