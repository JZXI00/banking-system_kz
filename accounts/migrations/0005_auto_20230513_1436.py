# Generated by Django 3.2 on 2023-05-13 08:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0004_userbankaccountclusters'),
    ]

    operations = [
        migrations.CreateModel(
            name='UserBankAccountProducts',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('product_code', models.CharField(blank=True, max_length=100, null=True)),
                ('product_name', models.CharField(blank=True, max_length=100, null=True)),
                ('description', models.TextField(blank=True, null=True)),
            ],
        ),
        migrations.AddField(
            model_name='userbankaccountclusters',
            name='cluster_code',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='userbankaccountclusters',
            name='description',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='userbankaccountclusters',
            name='cluster_name',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
    ]
