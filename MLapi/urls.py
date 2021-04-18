from django.contrib import admin
from django.urls import path,include
from core.views import CompasRaceDistribution,CompasGenderDistribution,CompasMLoperations,GermanGenderDistribution,GermanBadAndGoodDistribution,GermanMLoperations



urlpatterns = [
    path('api-auth/', include('rest_framework.urls')),
    path('admin/', admin.site.urls),
    path('Compas/Race',CompasRaceDistribution.as_view(),name= 'race'),
    path('Compas/Gender',CompasGenderDistribution.as_view(),name= 'gender'),
    path('Compas/Ml',CompasMLoperations.as_view(),name='machineLearning'),
    path('credit-risk/gender',GermanGenderDistribution.as_view(),name= 'german-gender'),
    path('credit-risk/bad-good',GermanBadAndGoodDistribution.as_view(),name= 'german-badgood'),
    path('German/Ml',GermanMLoperations.as_view(),name='GermanmachineLearning'),
]
