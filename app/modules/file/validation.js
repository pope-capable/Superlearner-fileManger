import Joi from '@hapi/joi';

export default {
  create: {
    body : {
      schema : Joi.object({
        name: Joi.string()
          .max(100)
          .trim()
          .required(),
        description: Joi.string()
      }),
    },
    params : {
      schema : Joi.object({
        folderId : Joi.number().integer().required()
      })
    }
  },

  update: {
    body : {
      schema : Joi.object({
        name: Joi.string()
          .max(100)
          .trim()
          .required(),
        description: Joi.string()
      }),
    },
    params : {
      schema : Joi.object({
        id : Joi.number().integer().required()
      })
    }
  },

  view: {
    params : {
      schema : Joi.object({
        id: Joi.number().integer().required()
      })
    }
  },

  list: {
    params : {
      schema : Joi.object({
        folderId : Joi.number().integer().required()
      })
    },
    query : {
      schema : Joi.object({
        pageId: Joi.number().integer().default(1),
        limit : Joi.number().integer().required()
      })
    }
  },

  remove: {
    params : {
      schema : Joi.object({
        id : Joi.number().integer().required()
      })
    }
  },

};
