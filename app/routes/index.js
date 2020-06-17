'use strict';
import Folder from '../modules/folder';
import File from '../modules/file';
import PredefinedFolder from '../modules/predefinedFolder';

export default app => {
  const version = '/api/' + process.env.API_VERSION;
  app.use(`${version}/`, Folder);
  app.use(`${version}/`, File);
  app.use(`${version}/`, PredefinedFolder);

  app.use((req, res, next) => {
    res.status(404).json({ message: 'Requested route not found' });
  });
};
