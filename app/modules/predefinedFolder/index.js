import { Router } from 'express';
import * as controller from './controller';

const route = Router();

route.get('/folders/seed', controller.seed);
route.get('/folders', controller.list);

export default route;
