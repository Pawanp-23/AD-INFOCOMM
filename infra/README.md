# infra/

Deployment and operations assets. Spec: [SAD §9](../docs/03-System-Architecture.md#9-deployment-architecture).

```
infra/
├── docker/        # backend.Dockerfile (API + worker), frontend.Dockerfile
├── nginx/         # reverse proxy: TLS, WebSocket upgrade, upload size limits
└── monitoring/    # prometheus.yml, Grafana dashboards
```

The root `docker-compose.yml` wires these together for local and staging environments.
