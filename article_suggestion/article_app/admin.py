# admin.py
from django.contrib import admin
from .models import User

class UserAdmin(admin.ModelAdmin):
    list_display = ('email', 'full_name', 'gender', 'birth_date', 'username')  # The fields to be displayed in the list view
    search_fields = ('email', 'full_name')  # The fields to be searched in the search box
    list_filter = ('gender',)  # The fields to be used as filters
    ordering = ('email',)  # The default sorting field(s)

    # The fields to be used in updates on admin site. 
    # If you want all fields to be editable, you can remove this line.
    fields = ('email', 'full_name', 'gender', 'birth_date', 'username', 'password', 'interests')

admin.site.register(User, UserAdmin)
admin.site.site_header = "Article Suggestions Admin"
admin.site.site_title = "Article Suggestions Admin Portal"
admin.site.index_title = "Welcome to Article Suggestions Admin Portal"