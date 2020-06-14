'use strict';
import { Router } from 'express';
import * as controller from './controller';
import { joiValidator } from 'iyasunday';
import validation from './validation';
// import { guard } from '../../utils';

const route = Router();

route.get('/folder/:id',joiValidator(validation.view), controller.view);
route.get('/folders/:pageId/:limit',joiValidator(validation.list), controller.list);
route.post('/folder',joiValidator(validation.create), controller.create);
route.patch('/folder/:id',joiValidator(validation.update), controller.update);
route.delete('/folder/:id',joiValidator(validation.remove), controller.remove);

export default route;
