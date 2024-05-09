from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager

class UserManager(BaseUserManager):
    def create_user(self, email, full_name, gender, birth_date, password=None):
        if not email:
            raise ValueError('Users must have an email address')

        user = self.model(
            email=self.normalize_email(email),
            full_name=full_name,
            gender=gender,
            birth_date=birth_date
        )

        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, full_name, gender, birth_date, password=None):
        user = self.create_user(
            email,
            full_name=full_name,
            gender=gender,
            birth_date=birth_date,
            password=password,
        )
        user.is_superuser = True
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
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    birth_date = models.DateField()
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)

    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['full_name', 'gender', 'birth_date']

    def __str__(self):
        return self.email
    
    def has_perm(self, perm, obj=None):
        return True
    
    def has_module_perms(self, app_label):
        return True
    

