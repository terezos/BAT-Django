from django.contrib import admin
from django.urls import path,include
from core.views import RaceDistribution,GenderDistribution,MLoperations



urlpatterns = [
    path('api-auth/', include('rest_framework.urls')),
    path('admin/', admin.site.urls),
    path('Compas/Race',RaceDistribution.as_view(),name= 'race'),
    path('Compas/Gender',GenderDistribution.as_view(),name= 'gender'),
    path('Compas/Ml',MLoperations.as_view(),name='machineLearning')
]
