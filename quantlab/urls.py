from django.contrib import admin
from django.urls import include, path

admin.site.site_header = "QuantLab admin"
admin.site.site_title = "QuantLab admin"
admin.site.index_title = "Local research database"

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("desk.urls")),
]
