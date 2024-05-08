from django.db import models

class User(models.Model):
    GENDER_CHOICES = [
            ('M', 'Male'),
            ('F', 'Female'),
            ('O', 'Other'),
        ]

    full_name = models.CharField(max_length=50)

    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    birth_date = models.DateField()

    def __str__(self):
        return self.first_name + " " + self.last_name    
    

