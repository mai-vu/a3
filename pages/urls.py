from django.urls import path
from .views import homePageView, evalCar

urlpatterns = [
    path('', homePageView, name='home'),
    path('evalCar/', evalCar, name='evalCar'),
]

