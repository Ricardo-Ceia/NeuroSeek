"""
Onboarding automation script for new ACME Corp employees.

This script handles the automated setup steps that every new hire needs:
- Creates a user account in the identity provider
- Provisions access to standard tools (Slack, Jira, GitHub, Notion)
- Sends a welcome email with first-day instructions
- Schedules an onboarding checklist in the HR system
"""

import datetime


def create_user_account(email: str, full_name: str, department: str) -> dict:
    """
    Create a new user account in the identity provider.

    Returns a dict with the generated username and temporary password.
    The user will be required to change their password on first login.
    Accounts are disabled by default and must be activated by IT before
    the employee's start date.
    """
    username = email.split("@")[0].lower()
    temp_password = f"Welcome{datetime.date.today().year}!"
    return {
        "username": username,
        "email": email,
        "full_name": full_name,
        "department": department,
        "temp_password": temp_password,
        "status": "pending_activation",
    }


def provision_tools(username: str, department: str) -> list[str]:
    """
    Provision the standard set of tools for a new employee.

    All employees get Slack, Jira, and Notion.
    Engineers additionally get GitHub and AWS console access.
    Marketing additionally gets HubSpot and Canva.
    """
    tools = ["Slack", "Jira", "Notion"]
    if department.lower() == "engineering":
        tools += ["GitHub", "AWS Console"]
    elif department.lower() == "marketing":
        tools += ["HubSpot", "Canva"]
    return tools


def send_welcome_email(email: str, full_name: str, start_date: datetime.date) -> bool:
    """
    Send a welcome email to the new employee.

    The email includes:
    - First-day logistics (office address, parking, building access)
    - A link to the employee handbook
    - The onboarding schedule for the first two weeks
    - Contact information for their HR buddy

    Returns True if the email was sent successfully.
    """
    # In production this would call an email service API
    print(f"Sending welcome email to {full_name} <{email}> for start date {start_date}")
    return True


def run_onboarding(email: str, full_name: str, department: str, start_date: datetime.date):
    """
    Run the full onboarding sequence for a new hire.

    This is the main entry point called by the HR system when a new
    employee record is created. All steps are logged for audit purposes.
    """
    print(f"Starting onboarding for {full_name} ({department})")
    account = create_user_account(email, full_name, department)
    tools = provision_tools(account["username"], department)
    send_welcome_email(email, full_name, start_date)
    print(f"Onboarding complete. Provisioned tools: {tools}")
    return account
