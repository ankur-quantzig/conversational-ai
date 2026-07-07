@description('Azure region')
param location string = resourceGroup().location

@description('Container image for the FastAPI API')
param apiImage string

@description('Container image for the nginx/React frontend')
param frontendImage string

@secure()
@description('OpenAI API key')
param openaiApiKey string

@description('Azure Document Intelligence endpoint')
param documentIntelligenceEndpoint string

@secure()
@description('Azure Document Intelligence key')
param documentIntelligenceKey string

@secure()
@description('PostgreSQL database URL')
param databaseUrl string

@secure()
@description('JSON map of API keys to user/tenant/role metadata')
param appApiKeys string

@description('Allowed browser origins, comma separated')
param corsOrigins string

@description('Monthly budget amount for this resource group. Set to 0 to skip budget creation.')
param monthlyBudgetAmount int = 10

@description('Email address for Azure budget alerts')
param budgetAlertEmail string = 'ankurkumarj@quantzig.com'

@description('UTC budget start date in yyyy-mm-dd format')
param budgetStartDate string = utcNow('yyyy-MM-01')

@description('API minimum replicas. 0 enables Container Apps scale-to-zero on the consumption plan.')
@minValue(0)
@maxValue(1)
param apiMinReplicas int = 0

@description('API maximum replicas. Keep low while using free/credit-backed budgets.')
@minValue(1)
@maxValue(10)
param apiMaxReplicas int = 1

@description('Frontend minimum replicas. 0 enables scale-to-zero with cold starts.')
@minValue(0)
@maxValue(1)
param frontendMinReplicas int = 0

@description('Frontend maximum replicas. Keep low while using free/credit-backed budgets.')
@minValue(1)
@maxValue(10)
param frontendMaxReplicas int = 1

@description('Question limit for basic users')
param basicUserQuestionLimit string = '10'

@description('Comma-separated power user emails')
param powerUsers string = 'ankurkumarj@quantzig.com'

@description('Comma-separated basic user emails')
param basicUsers string = 'surajc@quantzig.com,sidhus@quantzig.com,akshatameemamshi@quantzig.com,vikasgoyal@quantzig.com,saiprasad@quantzig.com'

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'insight-copilot-logs'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'insight-copilot-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

resource api 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'insight-copilot-api'
  location: location
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      ingress: {
        external: false
        targetPort: 8000
      }
      secrets: [
        { name: 'openai-api-key', value: openaiApiKey }
        { name: 'document-intelligence-key', value: documentIntelligenceKey }
        { name: 'database-url', value: databaseUrl }
        { name: 'app-api-keys', value: appApiKeys }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: apiImage
          env: [
            { name: 'APP_ENV', value: 'production' }
            { name: 'CORS_ORIGINS', value: corsOrigins }
            { name: 'OPENAI_API_KEY', secretRef: 'openai-api-key' }
            { name: 'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT', value: documentIntelligenceEndpoint }
            { name: 'AZURE_DOCUMENT_INTELLIGENCE_KEY', secretRef: 'document-intelligence-key' }
            { name: 'DATABASE_URL', secretRef: 'database-url' }
            { name: 'APP_API_KEYS', secretRef: 'app-api-keys' }
            { name: 'OPENAI_GUARDRAIL_MODEL', value: 'gpt-4.1-mini' }
            { name: 'GUARDRAIL_CONFIDENCE_THRESHOLD', value: '0.9' }
            { name: 'RATE_LIMIT_PER_MINUTE', value: '30' }
            { name: 'BASIC_USER_QUESTION_LIMIT', value: basicUserQuestionLimit }
            { name: 'POWER_USERS', value: powerUsers }
            { name: 'BASIC_USERS', value: basicUsers }
          ]
        }
      ]
      scale: {
        minReplicas: apiMinReplicas
        maxReplicas: apiMaxReplicas
      }
    }
  }
}

resource frontend 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'insight-copilot-frontend'
  location: location
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      ingress: {
        external: true
        targetPort: 80
        transport: 'auto'
      }
    }
    template: {
      containers: [
        {
          name: 'frontend'
          image: frontendImage
        }
      ]
      scale: {
        minReplicas: frontendMinReplicas
        maxReplicas: frontendMaxReplicas
      }
    }
  }
}

resource monthlyBudget 'Microsoft.Consumption/budgets@2023-05-01' = if (monthlyBudgetAmount > 0) {
  name: 'insight-copilot-monthly-budget'
  properties: {
    category: 'Cost'
    amount: monthlyBudgetAmount
    timeGrain: 'Monthly'
    timePeriod: {
      startDate: '${budgetStartDate}T00:00:00Z'
      endDate: '2036-12-31T00:00:00Z'
    }
    notifications: {
      actual80: {
        enabled: true
        operator: 'GreaterThan'
        threshold: 80
        contactEmails: [
          budgetAlertEmail
        ]
      }
      actual100: {
        enabled: true
        operator: 'GreaterThan'
        threshold: 100
        contactEmails: [
          budgetAlertEmail
        ]
      }
      forecast100: {
        enabled: true
        operator: 'GreaterThan'
        threshold: 100
        thresholdType: 'Forecasted'
        contactEmails: [
          budgetAlertEmail
        ]
      }
    }
  }
}

output frontendUrl string = 'https://${frontend.properties.configuration.ingress.fqdn}'
