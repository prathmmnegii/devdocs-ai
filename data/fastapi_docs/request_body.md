
# Request Bodies

A request body contains data sent by a client to an API.

FastAPI commonly uses Pydantic models to define the
structure and types of request body data.

When a Pydantic model is declared as a parameter in
an endpoint, FastAPI validates the incoming JSON data.

Invalid request data produces a validation error.
