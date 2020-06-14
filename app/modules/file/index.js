'use strict';
import { Router } from 'express';
import * as controller from './controller';
import { joiValidator, uploadFile } from 'iyasunday';
import validation from './validation';
import { FILE_FORMAT } from '../../utils/constant';

const route = Router();

route.get('/file/:id',joiValidator(validation.view), controller.view);
route.get('/folder/:folderId/files',joiValidator(validation.list), controller.list);

route.post('/folder/:folderId/file',uploadFile({
    location : process.env.STORAGE_PATH+'/uploads',
    allowedFormat : FILE_FORMAT
}).single('file'),joiValidator(validation.create), controller.create);

route.patch('/file/:id',uploadFile({
    location : process.env.STORAGE_PATH+'/uploads',
    allowedFormat : FILE_FORMAT
}).single('file'),joiValidator(validation.update), controller.update);

route.delete('/file/:id',joiValidator(validation.remove), controller.remove);

export default route;
