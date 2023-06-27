from cpppo.server.enip import server
from cpppo.server.enip import datastore

# Create the server data store
data_store = datastore.DictionaryDataStore()

# Set the value of "my_variable" to 8
data_store['my_variable'] = 8

# Create the server
server.create_server(address=('192.168.1.20', 44818), data_store=data_store)
