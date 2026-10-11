
# Path Parameters

Path parameters are variables included in a URL path.

For example, /items/{item_id} defines a path parameter
named item_id.

FastAPI can convert path parameters to declared Python
types using type annotations.

If item_id is declared as an integer, FastAPI validates
the incoming value and converts it to an integer.
