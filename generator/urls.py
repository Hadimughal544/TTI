from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('generate/', views.generate, name='generate'),
    path('delete/<int:image_id>/', views.delete_image, name='delete_image'),
]
