# server/leads.py
# 5 diverse B2B lead profiles used across all 3 tasks.
# Each lead has rich context to test personalization quality.

LEADS = [
    {
        "id": "lead_001",
        "name": "Priya Sharma",
        "title": "Head of Engineering",
        "company": "FinStack Technologies",
        "industry": "Fintech",
        "company_size": "150 employees",
        "pain_points": ["scaling engineering team", "technical debt", "slow deployments"],
        "recent_news": "FinStack just raised Series B of $20M",
        "tech_stack": ["Python", "AWS", "Kubernetes"],
        "linkedin_activity": "Recently posted about challenges in microservices migration",
        "persona": "busy, data-driven, skeptical of cold outreach",
    },
    {
        "id": "lead_002",
        "name": "David Okafor",
        "title": "VP of Sales",
        "company": "RetailLoop Inc",
        "industry": "E-commerce SaaS",
        "company_size": "80 employees",
        "pain_points": ["low demo conversion", "long sales cycles", "CRM data quality"],
        "recent_news": "RetailLoop launched a new enterprise tier last quarter",
        "tech_stack": ["Salesforce", "HubSpot", "Intercom"],
        "linkedin_activity": "Shared an article on reducing churn in B2B SaaS",
        "persona": "results-oriented, responds to ROI and numbers",
    },
    {
        "id": "lead_003",
        "name": "Anjali Mehta",
        "title": "Chief Operations Officer",
        "company": "MedBridge Health",
        "industry": "HealthTech",
        "company_size": "300 employees",
        "pain_points": ["manual reporting workflows", "compliance overhead", "staff burnout"],
        "recent_news": "MedBridge expanded to 3 new states this year",
        "tech_stack": ["Epic EHR", "Tableau", "Slack"],
        "linkedin_activity": "Posted about operational efficiency in healthcare",
        "persona": "conservative, values trust and security, risk-averse",
    },
    {
        "id": "lead_004",
        "name": "Marcus Webb",
        "title": "Founder & CEO",
        "company": "SupplyNest",
        "industry": "Supply Chain SaaS",
        "company_size": "25 employees",
        "pain_points": ["inventory forecasting", "supplier communication delays", "cash flow"],
        "recent_news": "SupplyNest was featured in TechCrunch last month",
        "tech_stack": ["Shopify", "QuickBooks", "Airtable"],
        "linkedin_activity": "Asked for recommendations on demand planning tools",
        "persona": "entrepreneurial, quick decision-maker, tight budget",
    },
    {
        "id": "lead_005",
        "name": "Lena Fischer",
        "title": "Director of People & Culture",
        "company": "Klarity GmbH",
        "industry": "HR Tech",
        "company_size": "200 employees",
        "pain_points": ["employee retention", "performance review process", "onboarding time"],
        "recent_news": "Klarity is hiring aggressively with 40 open positions",
        "tech_stack": ["Workday", "Notion", "Zoom"],
        "linkedin_activity": "Published a post on remote onboarding challenges",
        "persona": "empathetic, people-first, values culture fit",
    },
]

# 4 realistic objection types used in Task 3 (Hard)
OBJECTIONS = [
    {
        "type": "timing",
        "response": "Thanks for reaching out but this really isn't a good time. We're in the middle of a product launch.",
        "recovery_keywords": ["understand", "brief", "when would", "right time", "later", "quick"],
    },
    {
        "type": "budget",
        "response": "We've already spent our budget for this quarter. Can't take on anything new right now.",
        "recovery_keywords": ["ROI", "saves", "cost", "free trial", "next quarter", "invest", "return"],
    },
    {
        "type": "competitor",
        "response": "We're already using a competitor solution and pretty happy with it.",
        "recovery_keywords": ["different", "compare", "specifically", "unique", "unlike", "instead", "additionally"],
    },
    {
        "type": "not_relevant",
        "response": "I'm not sure this is relevant to what we actually do.",
        "recovery_keywords": ["specifically", "your industry", "companies like", "exactly", "similar", "relevant"],
    },
]