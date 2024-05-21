from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
from djongo import models as djongo_models

class UserManager(BaseUserManager):
    def create_user(self, full_name, gender, birth_date, username, password=None, interests=None, email="admin@gmail.com"):
        if not email:
            email = "admin@gmail.com"

        user = self.model(
            email=self.normalize_email(email),
            username = username,
            full_name=full_name,
            gender=gender,
            birth_date=birth_date,
            interests=interests or [],
        )
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, username, full_name, gender, birth_date, password=None):
        user = self.create_user(
            username = username,
            full_name=full_name,
            gender=gender,
            birth_date=birth_date,
            password=password,
        )
        user.email = "gurkan@gmail.com"
        user.is_superuser = True
        user.is_staff = True  # Add this line
        user.save(using=self._db)
        return user
    
class User(AbstractBaseUser):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    email = models.EmailField(
        verbose_name='email address',
        max_length=255,
        unique=True,
    )
    
    full_name = models.CharField(max_length=50)
    username = models.CharField(max_length=100, unique=True)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    birth_date = models.DateField()
    interests = djongo_models.JSONField(default=list)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)

    objects = UserManager()

    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = ['full_name', 'gender', 'birth_date']

    def __str__(self):
        return self.email
    
    def has_perm(self, perm, obj=None):
        return True
    
    def has_module_perms(self, app_label):
        return True
    

    

