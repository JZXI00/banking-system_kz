from django.contrib import admin

from .models import BankAccountType, User, UserAddress, UserBankAccount, UserBankAccountCluster, UserBankAccountProduct


@admin.register(UserBankAccount)
class UserBankAccountAdmin(admin.ModelAdmin):
    list_display = ('account_no', 'user', 'birth_date', 'balance', 'cluster_label', 'product', 'is_churned')

@admin.register(UserBankAccountCluster)
class UserBankAccountClusterAdmin(admin.ModelAdmin):
    list_display = ('cluster_code', 'cluster_name', 'description')

@admin.register(UserBankAccountProduct)
class UserBankAccountProductAdmin(admin.ModelAdmin):
    list_display = ('product_code', 'product_name', 'description')

admin.site.register(BankAccountType)
admin.site.register(User)
admin.site.register(UserAddress)

