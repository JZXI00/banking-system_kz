# Generated by Django 3.2.9 on 2023-05-09 15:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('transactions', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='transaction',
            name='transaction_type',
            field=models.PositiveSmallIntegerField(choices=[(1, 'ДЕПОЗИТ'), (2, 'АУДАРУ'), (3, 'ПАЙЫЗ')]),
        ),
    ]