# Job API

The **Dataspace Job API** is the entrypoint for the **EDC Connector** and **other services** into the **EDP Service**. It allows uploading data, starting the analysis job that is executed by the **EDPS Analysis Backend** and finally downloading the processing results. In this document we show the components involved and the steps to go through for an API user.

For information on how to start the Job REST API refer to [the scripts](./../../scripts/README.md#run-docker-containers).

## Components

This is how the components work together on a high-level:

```mermaid
flowchart TD
    EDC[EDC Connector] -->|Sends data for analysis,<br/>reads back computed EDP| API
    subgraph "EDPS System"
        API[Dataspace Job API] -->|Controls the<br/>analysis run| ANALYSIS[EDPS Analysis Backend]
    end
```

- EDC Connector: An external datahub uploads assets for analysis to the EDPS system and downloads the processing results (computed **EDP** with plots and PDF report).
- Dataspace Job API: Entrypoint for controlling analysis jobs. The API is specified in **OpenAPI**, supporting **SwaggerUI** and **ReDoc** among other tools.
- EDPS Analysis Backend: Processes the data and provides the results to the API.

## Processing steps

```mermaid
sequenceDiagram
    participant EDC as EDC Connector
    participant API as Dataspace Job API
    participant ANALYSIS as EDPS Analysis Backend

    rect rgb(235,235,235)
        EDC->>API: Create analysis job [User-metadata]
        activate API
        API->>EDC: [Job-ID, Status: WAITING_FOR_DATA]
        deactivate API
    end

    rect rgb(235,235,235)
        EDC->>API: Upload input data [Job-ID, File]
        activate API
        API->>EDC: [Status: QUEUED]
        deactivate API
    end

    rect rgb(235,235,235)
        note over API,ANALYSIS: Background thread
        API->>ANALYSIS: Start analysis
        activate ANALYSIS
        ANALYSIS->>ANALYSIS: Processing
        ANALYSIS->>API: Resulting EDP with plots and PDF report
        deactivate ANALYSIS
        activate API
        API->>API: Zip results
        deactivate API
    end

    rect rgb(235,235,235)
        EDC->>API : Poll job status [Job-ID]
        activate API
        API->>EDC: [Status: COMPLETED]
    end

    rect rgb(235,235,235)
        EDC->>API : Download job results [Job-ID]
        activate API
        API->>EDC: [ZIP: EDP with plots and PDF report]
    end
```

### Create job

First the **EDC Connector** creates a new analysis job by uploading the user-metadata to the API. The API returns the **job info** containing the randomly generated **Job-ID** and status (at this point **WAITING_FOR_DATA**).

You can specify the "x-service-base-url" header field in this call. That will lead to the response containing an extra field "upload_url", which contains "{x_service_base_url}/api/{job_id}".

### Upload data

Referring to the new job using **Job-ID** the caller uploads the raw input data (e.g. a CSV file) and provides its filename. The API again returns the **job info**, the status now being **QUEUED**.

### Process analysis job 

The API starts the analysis in a background process, changing the status to **PROCESSING**.
If the analysis was successful the results consisting of the computed EDP, plots and the PDF report are zipped and the status changes to **COMPLETED**.
In case of an error the status changes to **FAILED** and the error message is stored.

### Poll status

The client should poll for status updates passing the **Job-ID** until it reaches **COMPLETED** or **FAILED**. For a failed job an error message is returned.

### Download results

If the job reaches status **COMPLETED** the client can download the resulting ZIP file after passing the **Job-ID**.
