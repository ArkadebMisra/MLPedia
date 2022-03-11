from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    #log in log out view
    path('login/', auth_views.LoginView.as_view(), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),

    # change password urls
    path('password_change/',
            auth_views.PasswordChangeView.as_view(),
            name='password_change'),
    path('password_change/done/',
            auth_views.PasswordChangeDoneView.as_view(),
            name='password_change_done'),

    #user registration url
    path('register/', views.register, name='register'),
    
    #edit profile url
    path('edit/', views.edit, name='edit'),
]