# IOT-Agent
Real time video recorder from live camera streams.

## Setup

Set credentials in .env file:
```bash
AZURE_ACCOUNT_URL=https://cienciaciudades2024.blob.core.windows.net
AZURE_STORAGE_SAS_TOKEN=sp=racwdli&st=2025-05-24T00:26:16Z&se=2026-10-09T08:26:16Z&sv=2024-11-04&sr=c&sig=4i9KPAZkChsYcadv%2BdNg1ZDusltPVOg3DG3tbJ5t9v4%3D
```

Set stream links on [config file](./Public/config/system.release.standard.config.ini).

## Running

Run the following command to start the IOT-Agent:
```bash
cd docker compose up iot-agent-main -d
```

Logs are saved at for each run at their own [folder](./Public/log/branch/)