'use strict';
import helmet from 'helmet';
import compression from 'compression';
import bodyParser from 'body-parser';
import cors from 'cors';

export default app => {
  if (process.env.NODE_ENV === 'production') {
    app.use(compression());
    app.use(helmet());
  }

  app.use(bodyParser.json());
  app.use(bodyParser.urlencoded({ extended: true }));
  app.use(cors());

  if (process.env.NODE_ENV !== 'production') {
    app.use((req, res, next) => {
      console.log(req.body);
      console.log('=========================================');
      console.log({
        token: req.headers.authorization,
        method: req.method,
        url: `${req.get('HOST')}${req.originalUrl}`,
        body: req.body,
        params: req.params,
        query: req.query,
      });
      console.log('=========================================');
      next();
    });
  }
};
