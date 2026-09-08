from django.urls import path

from . import views

app_name = "desk"
urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("portfolios/<int:pk>/", views.portfolio_detail, name="portfolio_detail"),
    path("<str:page>/", views.workspace_page, name="page"),
    path("portfolios/create/", views.create_portfolio, name="portfolio_create"),
    path(
        "portfolios/create-manual/",
        views.create_manual_portfolio,
        name="portfolio_create_manual",
    ),
    path("portfolios/<int:pk>/edit/", views.edit_portfolio, name="portfolio_edit"),
    path(
        "portfolios/<int:pk>/clone/",
        views.clone_saved_portfolio,
        name="portfolio_clone",
    ),
    path(
        "portfolios/<int:pk>/archive/",
        views.archive_portfolio,
        name="portfolio_archive",
    ),
    path(
        "stock-selection/preview/",
        views.preview_stock_selection,
        name="stock_selection_preview",
    ),
    path(
        "stock-selection/save/",
        views.save_stock_selection_screen,
        name="stock_selection_save",
    ),
    path(
        "stock-selection/portfolio/",
        views.create_stock_selection_portfolio,
        name="stock_selection_portfolio",
    ),
    path("screens/create/", views.create_screen, name="screen_create"),
    path(
        "screens/runs/<int:pk>/portfolio/",
        views.create_portfolio_from_screen,
        name="screen_portfolio",
    ),
    path("signals/company/", views.open_signal_company, name="signal_company"),
    path(
        "factors/builds/<int:build_id>/<int:factor_id>/",
        views.factor_return_detail,
        name="factor_return_detail",
    ),
    path("jobs/launch/<str:kind>/", views.launch_domain_job, name="domain_job"),
    path("jobs/<int:pk>/cancel/", views.cancel_job, name="job_cancel"),
    path("jobs/<int:pk>/", views.job_status, name="job_status"),
    path("export/<str:kind>/<int:pk>.json", views.export_run_json, name="export_json"),
    path(
        "export/optimization/<int:pk>/holdings.csv",
        views.export_holdings_csv,
        name="export_holdings",
    ),
    path(
        "optimization/<int:pk>/backtest/",
        views.launch_scenario_backtest,
        name="scenario_backtest",
    ),
    path(
        "export/portfolio/<int:pk>/holdings.csv",
        views.export_portfolio_csv,
        name="export_portfolio",
    ),
]
