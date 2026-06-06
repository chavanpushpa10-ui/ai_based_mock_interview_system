from django.contrib import admin
from django.urls import path
from app import views

urlpatterns = [
    # path('',views.base, name="base"),
    path('',views.index, name='index'),
    path('about',views.about, name='about'),
    path('services',views.services, name='services'),
    path('contact',views.contact, name='contact'),
    path('log_in',views.log_in, name='log_in'),
    path('log_out',views.log_out, name='log_out'),
    path('register',views.register, name='register'),
    path('dashboard',views.dashboard, name='dashboard'),
    path('test',views.test, name='test'),
    path('resume',views.resume, name='resume'),
    path('user_intro',views.user_intro, name='user_intro'),
    path('user_test',views.user_test, name='user_test'),
    path('result',views.result, name='result'),
    path('adminlogin',views.adminlogin, name='adminlogin'),
    path('admindashboard',views.admindashboard,name='admindashboard'),
    path('delete_user/<int:user_id>/', views.delete_user, name='delete_user'),
    path('technicalinterview', views.technicalinterview, name='technicalinterview'),

]
