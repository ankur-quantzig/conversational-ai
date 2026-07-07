# Insight Copilot Deployment Hardening

## Do Not Deploy Local Secrets

- Do not commit `.env`.
- Rotate any keys that were ever placed in `.env` before creating a public URL.
- Store production secrets in Azure Container Apps secrets, Key Vault, or your CI/CD secret store.
- Do not deploy `output/chat_history.sqlite3`.
- Do not deploy local `data/` except approved static assets such as `data/logo`.

## Required Production Environment

Set:

```text
APP_ENV=production
CORS_ORIGINS=https://your-frontend-domain.example.com
APP_API_KEYS={...}
DATABASE_URL=postgresql://...
OPENAI_API_KEY=...
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=...
AZURE_DOCUMENT_INTELLIGENCE_KEY=...
OPENAI_GUARDRAIL_MODEL=gpt-4.1-mini
GUARDRAIL_CONFIDENCE_THRESHOLD=0.9
RATE_LIMIT_PER_MINUTE=30
BASIC_USER_QUESTION_LIMIT=10
POWER_USERS=ankurkumarj@quantzig.com
BASIC_USERS=surajc@quantzig.com,sidhus@quantzig.com,akshatameemamshi@quantzig.com,vikasgoyal@quantzig.com,saiprasad@quantzig.com
```

When `APP_ENV` is not local/dev/test:

- API-key auth is required.
- `CORS_ORIGINS` is required.
- Wildcard CORS is not used.

## Recommended Azure Shape

```text
Internet
  -> Azure Container Apps frontend
  -> FastAPI API container
  -> Azure Database for PostgreSQL
  -> Azure Key Vault / Container App secrets
  -> Log Analytics / Application Insights
```

## Final Pre-Deploy Checklist

1. Rotate OpenAI and Azure keys.
2. Create strong random `APP_API_KEYS` values for each user.
3. Set `APP_ENV=production`.
4. Set exact `CORS_ORIGINS`.
5. Use managed PostgreSQL, not SQLite.
6. Mount or package the approved `output/chunks`, `output/embeddings`, and `output/vector_db/lancedb` artifacts needed for retrieval.
7. Confirm `.env`, `output/chat_history.sqlite3`, `.venv/`, `node_modules/`, and unapproved generated data are not in the image or Git.
8. Build frontend with the production API URL.
9. Run a secret scan before pushing.
10. Test a malicious query and confirm `blocked_guardrail`.
11. Test audit, feedback, share, export, and retry endpoints.

## Current Auth Mode

This repo currently supports API-key auth through:

```text
Authorization: Bearer <api-key>
```

or:

```text
X-API-Key: <api-key>
```

The `APP_API_KEYS` env var is a JSON object:

```json
{
  "random-api-key": {
    "user_id": "ankurkumarj@quantzig.com",
    "email": "ankurkumarj@quantzig.com",
    "name": "Ankur Kumar",
    "tenant_id": "default",
    "roles": ["analyst", "power_user"],
    "document_ids": ["*"]
  }
}
```

Users in `POWER_USERS` have no question cap. All other users are treated as basic users and are limited by `BASIC_USER_QUESTION_LIMIT`, currently 10 questions. Keep the random API keys in Azure secrets or Key Vault, never in Git.

For enterprise production, replace or wrap this with Azure Entra ID/OIDC.

## Cost Controls

The Azure Bicep template uses cost-safe defaults:

- Azure Container Apps consumption environment.
- API `minReplicas=0` and `maxReplicas=1`.
- Frontend `minReplicas=0` and `maxReplicas=1`.
- A resource-group monthly budget named `insight-copilot-monthly-budget`.
- Budget notifications at 80% actual, 100% actual, and 100% forecasted spend.

Example deployment parameters:

```powershell
az account set --subscription "<subscription-id-with-credit>"

az group create `
  --name insight-copilot-rg `
  --location eastus

az deployment group create `
  --resource-group insight-copilot-rg `
  --template-file deploy/azure/container-apps.bicep `
  --parameters `
    apiImage="<your-api-image>" `
    frontendImage="<your-frontend-image>" `
    openaiApiKey="<secret>" `
    documentIntelligenceEndpoint="https://<your-resource>.cognitiveservices.azure.com/" `
    documentIntelligenceKey="<secret>" `
    databaseUrl="<postgres-url>" `
    appApiKeys="<json-api-key-map>" `
    corsOrigins="https://<your-frontend-domain>" `
    monthlyBudgetAmount=10 `
    budgetAlertEmail="ankurkumarj@quantzig.com"
```

Use the subscription that has the remaining Azure credit before running the deployment. The app cannot force Azure to spend from a specific credit pool from code; that is controlled by Azure subscription and billing setup.

You may need to do these items in the Azure portal:

1. Confirm the active subscription is the one with the remaining credit.
2. Make sure your deployment identity has permission to create budgets, usually Owner, Contributor plus Cost Management permissions, or Cost Management Contributor.
3. Confirm budget alert emails are allowed for `ankurkumarj@quantzig.com`.
4. Create a PostgreSQL free/trial-eligible server if your subscription offers it, or choose the smallest acceptable managed PostgreSQL SKU.
5. Add an Azure budget at the subscription level too, because the Bicep template budget is scoped to this app resource group only.

OpenAI API calls bill to the provider behind `OPENAI_API_KEY`. If you need model calls to draw from Azure billing/credit, use an Azure OpenAI deployment and update the app to call Azure OpenAI instead of the public OpenAI API.
